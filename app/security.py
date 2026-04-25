from __future__ import annotations

import hmac
import ipaddress
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque

from fastapi import status
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

HTTP_413 = getattr(status, "HTTP_413_CONTENT_TOO_LARGE", status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_csv(name: str) -> tuple[str, ...]:
    value = os.getenv(name, "")
    items = [item.strip() for item in value.split(",")]
    return tuple(item for item in items if item)


def _base64_char_limit(max_input_bytes: int) -> int:
    encoded = ((max_input_bytes + 2) // 3) * 4
    return encoded + 512


@dataclass(frozen=True)
class SecuritySettings:
    api_key: str | None
    allow_loopback_without_api_key: bool
    allowed_origins: tuple[str, ...]
    max_input_bytes: int
    max_base64_chars: int
    rate_limit_requests: int
    rate_limit_window_seconds: int
    max_concurrent_jobs: int


def load_security_settings() -> SecuritySettings:
    max_input_bytes = _env_int("MAX_INPUT_BYTES", 12_000_000)
    return SecuritySettings(
        api_key=os.getenv("ENHANCER_API_KEY") or None,
        allow_loopback_without_api_key=_env_bool("ALLOW_LOOPBACK_WITHOUT_API_KEY", True),
        allowed_origins=_env_csv("CORS_ALLOW_ORIGINS"),
        max_input_bytes=max_input_bytes,
        max_base64_chars=_base64_char_limit(max_input_bytes),
        rate_limit_requests=_env_int("RATE_LIMIT_REQUESTS", 20),
        rate_limit_window_seconds=_env_int("RATE_LIMIT_WINDOW_SECONDS", 60),
        max_concurrent_jobs=_env_int("MAX_CONCURRENT_JOBS", 2),
    )


class InMemoryRateLimiter:
    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = max(0, limit)
        self.window_seconds = max(1, window_seconds)
        self._buckets: defaultdict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        if self.limit <= 0:
            return True, 0

        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            bucket = self._buckets[key]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            if len(bucket) >= self.limit:
                retry_after = max(1, int(self.window_seconds - (now - bucket[0])))
                return False, retry_after

            bucket.append(now)
            return True, 0


class BodyTooLargeError(RuntimeError):
    """Raised when an HTTP request body exceeds the configured byte limit."""


class EdgeGuardMiddleware:
    def __init__(self, app: ASGIApp, settings: SecuritySettings, rate_limiter: InMemoryRateLimiter) -> None:
        self.app = app
        self.settings = settings
        self.rate_limiter = rate_limiter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "GET").upper()
        scheme = str(scope.get("scheme", "http")).lower()
        headers = Headers(scope=scope)

        if self._should_block_range(path, headers):
            response = PlainTextResponse("Byte range requests are disabled for static assets.", status_code=416)
            self._apply_security_headers(response.headers, headers, scheme)
            await response(scope, receive, send)
            return

        if method != "OPTIONS" and self._is_protected_api(path):
            failure = self._authorize(headers, scope)
            if failure is not None:
                self._apply_security_headers(failure.headers, headers, scheme)
                await failure(scope, receive, send)
                return

            allowed, retry_after = self.rate_limiter.allow(self._rate_limit_key(scope))
            if not allowed:
                response = JSONResponse(
                    {"detail": "Too many requests. Slow down and try again shortly."},
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    headers={"Retry-After": str(retry_after)},
                )
                self._apply_security_headers(response.headers, headers, scheme)
                await response(scope, receive, send)
                return

        if method != "OPTIONS" and self._is_body_limited_path(path):
            max_body_bytes = self._max_body_bytes_for_path(path)
            content_length = headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > max_body_bytes:
                        response = JSONResponse(
                            {"detail": self._body_limit_message(path)},
                            status_code=HTTP_413,
                        )
                        self._apply_security_headers(response.headers, headers, scheme)
                        await response(scope, receive, send)
                        return
                except ValueError:
                    pass

            receive = self._limited_receive(receive, max_body_bytes)

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                mutable_headers = MutableHeaders(scope=message)
                self._apply_security_headers(mutable_headers, headers, scheme)
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        except BodyTooLargeError:
            response = JSONResponse(
                {"detail": self._body_limit_message(path)},
                status_code=HTTP_413,
            )
            self._apply_security_headers(response.headers, headers, scheme)
            await response(scope, receive, send_with_headers)

    def _should_block_range(self, path: str, headers: Headers) -> bool:
        return "range" in headers and (path == "/" or path.startswith("/static/"))

    def _is_protected_api(self, path: str) -> bool:
        return path in {"/api/enhance", "/api/enhance/base64", "/api/engines"}

    def _is_body_limited_path(self, path: str) -> bool:
        return path in {"/api/enhance", "/api/enhance/base64"}

    def _authorize(self, headers: Headers, scope: Scope) -> Response | None:
        if self.settings.api_key:
            presented = headers.get("x-api-key") or self._bearer_token(headers)
            if presented and hmac.compare_digest(presented, self.settings.api_key):
                return None

            return JSONResponse(
                {"detail": "Provide a valid API key to access this endpoint."},
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": "Bearer"},
            )

        if self.settings.allow_loopback_without_api_key and self._is_loopback_request(headers, scope):
            return None

        return JSONResponse(
            {"detail": "API access is limited to loopback unless ENHANCER_API_KEY is configured."},
            status_code=status.HTTP_403_FORBIDDEN,
        )

    def _limited_receive(self, receive: Receive, max_body_bytes: int) -> Receive:
        total = 0

        async def inner() -> Message:
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > max_body_bytes:
                    raise BodyTooLargeError
            return message

        return inner

    def _rate_limit_key(self, scope: Scope) -> str:
        client = scope.get("client")
        if client and client[0]:
            return client[0]
        return "unknown"

    def _is_loopback_request(self, headers: Headers, scope: Scope) -> bool:
        client = scope.get("client")
        client_host = client[0] if client else None
        host_name = _host_without_port(headers.get("host", ""))
        return _is_loopback_host(client_host) and _is_loopback_host(host_name)

    def _bearer_token(self, headers: Headers) -> str | None:
        authorization = headers.get("authorization", "")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer":
            return None
        return token.strip() or None

    def _apply_security_headers(
        self,
        response_headers: MutableHeaders,
        request_headers: Headers,
        request_scheme: str,
    ) -> None:
        response_headers.setdefault("Content-Security-Policy", _content_security_policy())
        response_headers.setdefault("Referrer-Policy", "no-referrer")
        response_headers.setdefault("X-Content-Type-Options", "nosniff")
        response_headers.setdefault("X-Frame-Options", "DENY")
        response_headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
        response_headers.setdefault(
            "Permissions-Policy",
            "accelerometer=(), camera=(), geolocation=(), gyroscope=(), microphone=(), payment=(), usb=()",
        )
        if request_scheme == "https" or request_headers.get("x-forwarded-proto", "").lower() == "https":
            response_headers.setdefault("Strict-Transport-Security", "max-age=63072000; includeSubDomains")

    def _max_body_bytes_for_path(self, path: str) -> int:
        if path == "/api/enhance/base64":
            return self.settings.max_base64_chars + 2_048
        return self.settings.max_input_bytes

    def _body_limit_message(self, path: str) -> str:
        if path == "/api/enhance/base64":
            return "Base64 payload is too large."
        return f"Upload exceeds the {self.settings.max_input_bytes // 1_000_000} MB limit."


def _content_security_policy() -> str:
    return (
        "default-src 'self'; "
        "img-src 'self' data: blob:; "
        "style-src 'self'; "
        "script-src 'self'; "
        "connect-src 'self'; "
        "font-src 'self'; "
        "form-action 'self'; "
        "base-uri 'self'; "
        "frame-ancestors 'none'"
    )


def _is_loopback_host(value: str | None) -> bool:
    if not value:
        return False

    candidate = value.strip().lower()
    if candidate in {"localhost", "127.0.0.1", "::1"}:
        return True

    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def _host_without_port(value: str) -> str:
    if not value:
        return ""

    candidate = value.strip()
    if candidate.startswith("[") and "]" in candidate:
        return candidate[1 : candidate.index("]")]

    if candidate.count(":") == 1:
        host, _, _ = candidate.partition(":")
        return host

    return candidate
