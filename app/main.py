from __future__ import annotations

import base64
from binascii import Error as BinasciiError

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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

app = FastAPI(title="Product Image Enhancer", version="0.1.0")
enhancer = ProductImageEnhancer()
MAX_BASE64_CHARS = 40_000_000


class Base64EnhanceRequest(BaseModel):
    image_base64: str = Field(
        ...,
        description="Raw base64 image string or a data:image/...;base64,... URL.",
        min_length=1,
        max_length=MAX_BASE64_CHARS,
    )
    preset: PresetName = "product_standard"
    engine: EngineName = "pillow_fallback"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/engines")
def engines() -> dict[str, dict[str, object]]:
    return enhancer.engine_status()


@app.post("/api/enhance")
async def enhance_image(
    image: UploadFile = File(...),
    preset: PresetName = Form("product_standard"),
    engine: EngineName = Form("pillow_fallback"),
) -> dict[str, object]:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a valid image file.")

    file_bytes = await image.read()
    return _enhance_bytes(file_bytes, preset=preset, engine=engine)


@app.post("/api/enhance/base64")
def enhance_base64_image(payload: Base64EnhanceRequest) -> dict[str, object]:
    file_bytes = _decode_base64_image(payload.image_base64)
    return _enhance_bytes(file_bytes, preset=payload.preset, engine=payload.engine)


def _enhance_bytes(file_bytes: bytes, preset: PresetName, engine: EngineName) -> dict[str, object]:
    try:
        result = enhancer.enhance(file_bytes, preset=preset, engine=engine)
    except ImageEnhancementError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelRuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _result_to_response(result)


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


def _result_to_response(result: EnhancementResult) -> dict[str, object]:
    encoded = base64.b64encode(result.image_bytes).decode("ascii")
    return {
        "image": f"data:{result.content_type};base64,{encoded}",
        "width": result.width,
        "height": result.height,
        "engine": result.engine,
        "engine_label": result.engine_label,
        "preset": result.preset,
    }
