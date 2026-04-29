from __future__ import annotations

import asyncio
import base64
from binascii import Error as BinasciiError
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.enhancer import (
    EngineName,
    EnhancementResult,
    ImageEnhancementError,
    ModelRuntimeError,
    ModelUnavailableError,
    PresetName,
    ProductImageEnhancer,
)
from app.security import EdgeGuardMiddleware, InMemoryRateLimiter, SecuritySettings, load_security_settings


def create_app(security_settings: SecuritySettings | None = None) -> FastAPI:
    settings = security_settings or load_security_settings()
    enhancer = ProductImageEnhancer(max_input_bytes=settings.max_input_bytes)
    app = FastAPI(title="Product Image Enhancer", version="0.2.0")

    app.state.settings = settings
    app.state.enhancer = enhancer
    app.state.enhancement_slots = asyncio.Semaphore(max(1, settings.max_concurrent_jobs))

    if settings.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.allowed_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key"],
        )

    app.add_middleware(
        EdgeGuardMiddleware,
        settings=settings,
        rate_limiter=InMemoryRateLimiter(settings.rate_limit_requests, settings.rate_limit_window_seconds),
    )

    class Base64EnhanceRequest(BaseModel):
        image_base64: str = Field(
            ...,
            description="Raw base64 image string or a data:image/...;base64,... URL.",
            min_length=1,
            max_length=settings.max_base64_chars,
        )
        preset: PresetName = "product_standard"
        engine: EngineName = "pillow_fallback"

    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/")
    def index() -> Response:
        if settings.demo_as_root:
            return RedirectResponse(url="/demo/ai-image-enhancer", status_code=307)
        return FileResponse("static/index.html")

    @app.get("/demo/ai-image-enhancer")
    def demo_ai_image_enhancer() -> FileResponse:
        return FileResponse("static/ai-image-enhancer-demo.html")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/engines")
    def engines() -> dict[str, dict[str, object]]:
        return enhancer.engine_status()

    @app.post("/api/enhance")
    async def enhance_image(
        request: Request,
        preset: PresetName = Query("product_standard"),
        engine: EngineName = Query("pillow_fallback"),
        response_format: Literal["binary", "data_url"] = Query("binary"),
    ) -> Response:
        content_type = (request.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Send the uploaded file as a raw image request body.")

        file_bytes = await _read_request_bytes(request)
        result = await _enhance_bytes(enhancer, app.state.enhancement_slots, file_bytes, preset=preset, engine=engine)

        if response_format == "data_url":
            return _result_to_data_url_response(result)

        return _result_to_binary_response(result)

    @app.post("/demo/api/enhance")
    async def demo_enhance_image(request: Request) -> Response:
        content_type = (request.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Send the uploaded file as a raw image request body.")

        file_bytes = await _read_request_bytes(request)
        result = await _enhance_bytes(
            enhancer,
            app.state.enhancement_slots,
            file_bytes,
            preset="product_detail",
            engine="studio_product_focus",
        )
        return _result_to_binary_response(result)

    @app.post("/api/enhance/base64")
    async def enhance_base64_image(payload: Base64EnhanceRequest) -> JSONResponse:
        file_bytes = _decode_base64_image(payload.image_base64)
        if len(file_bytes) > settings.max_input_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Decoded image exceeds the {settings.max_input_bytes // 1_000_000} MB limit.",
            )
        result = await _enhance_bytes(
            enhancer,
            app.state.enhancement_slots,
            file_bytes,
            preset=payload.preset,
            engine=payload.engine,
        )
        return _result_to_data_url_response(result)

    return app


async def _read_request_bytes(request: Request) -> bytes:
    chunks: list[bytes] = []
    async for chunk in request.stream():
        if chunk:
            chunks.append(chunk)
    return b"".join(chunks)


async def _enhance_bytes(
    enhancer: ProductImageEnhancer,
    enhancement_slots: asyncio.Semaphore,
    file_bytes: bytes,
    preset: PresetName,
    engine: EngineName,
) -> EnhancementResult:
    try:
        async with enhancement_slots:
            return await run_in_threadpool(enhancer.enhance, file_bytes, preset, engine)
    except ImageEnhancementError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelRuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _decode_base64_image(image_base64: str) -> bytes:
    value = image_base64.strip()
    if not value:
        raise HTTPException(status_code=400, detail="Upload a base64 image before enhancing.")

    if value.startswith("data:"):
        header, separator, encoded = value.partition(",")
        header_lower = header.lower()
        if not separator or ";base64" not in header_lower:
            raise HTTPException(status_code=400, detail="Use a valid base64 data URL.")
        if not header_lower.startswith("data:image/"):
            raise HTTPException(status_code=400, detail="Use a data URL with an image media type.")
    else:
        encoded = value

    encoded = "".join(encoded.split())
    try:
        return base64.b64decode(encoded, validate=True)
    except (BinasciiError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Use a valid base64 image string.") from exc


def _result_to_binary_response(result: EnhancementResult) -> Response:
    return Response(
        content=result.image_bytes,
        media_type=result.content_type,
        headers={
            "Cache-Control": "no-store",
            "X-Image-Width": str(result.width),
            "X-Image-Height": str(result.height),
            "X-Enhancer-Engine": result.engine,
            "X-Enhancer-Engine-Label": result.engine_label,
            "X-Enhancer-Preset": result.preset,
        },
    )


def _result_to_data_url_payload(result: EnhancementResult) -> dict[str, object]:
    encoded = base64.b64encode(result.image_bytes).decode("ascii")
    return {
        "image": f"data:{result.content_type};base64,{encoded}",
        "width": result.width,
        "height": result.height,
        "engine": result.engine,
        "engine_label": result.engine_label,
        "preset": result.preset,
    }


def _result_to_data_url_response(result: EnhancementResult) -> JSONResponse:
    return JSONResponse(_result_to_data_url_payload(result), headers={"Cache-Control": "no-store"})


app = create_app()
