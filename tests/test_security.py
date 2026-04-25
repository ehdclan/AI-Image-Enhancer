from __future__ import annotations

import io
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.security import SecuritySettings


def _make_settings(**overrides) -> SecuritySettings:
    values = {
        "api_key": None,
        "allow_loopback_without_api_key": True,
        "allowed_origins": (),
        "max_input_bytes": 2_000_000,
        "max_base64_chars": 2_667_180,
        "rate_limit_requests": 20,
        "rate_limit_window_seconds": 60,
        "max_concurrent_jobs": 2,
    }
    values.update(overrides)
    return SecuritySettings(**values)


def _sample_png_bytes() -> bytes:
    image = Image.new("RGB", (32, 24), (18, 97, 73))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class SecurityPatchTests(unittest.TestCase):
    def test_loopback_can_check_engines_without_api_key(self) -> None:
        client = TestClient(
            create_app(_make_settings()),
            base_url="http://127.0.0.1",
            client=("127.0.0.1", 50000),
        )

        response = client.get("/api/engines")

        self.assertEqual(response.status_code, 200)
        self.assertIn("studio_product", response.json())

    def test_remote_engines_request_is_forbidden_without_api_key(self) -> None:
        client = TestClient(create_app(_make_settings()), base_url="http://example.com")

        response = client.get("/api/engines")

        self.assertEqual(response.status_code, 403)

    def test_api_key_unlocks_remote_access(self) -> None:
        client = TestClient(
            create_app(_make_settings(api_key="secret", allow_loopback_without_api_key=False)),
            base_url="http://example.com",
        )

        unauthorized = client.get("/api/engines")
        authorized = client.get("/api/engines", headers={"X-API-Key": "secret"})

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(authorized.status_code, 200)

    def test_raw_image_endpoint_returns_binary_image_response(self) -> None:
        client = TestClient(
            create_app(_make_settings()),
            base_url="http://127.0.0.1",
            client=("127.0.0.1", 50000),
        )

        response = client.post(
            "/api/enhance?engine=pillow_fallback&preset=product_standard",
            content=_sample_png_bytes(),
            headers={"Content-Type": "image/png"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.headers["x-enhancer-engine"], "pillow_fallback")
        self.assertGreater(len(response.content), 0)

    def test_oversized_request_is_rejected_before_processing(self) -> None:
        client = TestClient(
            create_app(_make_settings(max_input_bytes=64, max_base64_chars=256)),
            base_url="http://127.0.0.1",
            client=("127.0.0.1", 50000),
        )

        response = client.post(
            "/api/enhance?engine=pillow_fallback&preset=product_standard",
            content=b"x" * 512,
            headers={"Content-Type": "image/png"},
        )

        self.assertEqual(response.status_code, 413)

    def test_range_requests_are_blocked_for_static_assets(self) -> None:
        client = TestClient(create_app(_make_settings()), base_url="http://127.0.0.1")

        response = client.get("/static/app.js", headers={"Range": "bytes=0-8"})

        self.assertEqual(response.status_code, 416)


if __name__ == "__main__":
    unittest.main()
