from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)

PresetName = Literal["product_standard", "product_detail", "product_soft"]
EngineName = Literal["realesrgan", "pillow_fallback"]

REALESRGAN_MODEL_PATH = Path(os.getenv("REALESRGAN_MODEL_PATH", "weights/RealESRGAN_x4plus.pth"))


@dataclass(frozen=True)
class EnhancementResult:
    image_bytes: bytes
    content_type: str
    width: int
    height: int
    engine: EngineName
    engine_label: str
    preset: PresetName


class ImageEnhancementError(ValueError):
    """Raised when an uploaded file cannot be enhanced safely."""


class ModelUnavailableError(RuntimeError):
    """Raised when the selected enhancement engine is not configured."""


class ModelRuntimeError(RuntimeError):
    """Raised when a configured model fails during inference."""


class ProductImageEnhancer:
    """Conservative product-image enhancement pipeline.

    The production path is Real-ESRGAN when the optional dependency and weights
    are available. The built-in fallback is deterministic and intentionally mild,
    so demos and local development can run without changing product details.
    """

    def __init__(self, max_input_pixels: int = 24_000_000, target_long_edge: int = 1800) -> None:
        self.max_input_pixels = max_input_pixels
        self.target_long_edge = target_long_edge
        self._realesrgan_status = "Checking Real-ESRGAN runtime."
        self._realesrgan = self._load_realesrgan()

    def enhance(
        self,
        file_bytes: bytes,
        preset: PresetName = "product_standard",
        engine: EngineName = "pillow_fallback",
    ) -> EnhancementResult:
        image = self._decode(file_bytes)

        if engine == "realesrgan":
            if self._realesrgan is None:
                self._realesrgan = self._load_realesrgan()

            if self._realesrgan is None:
                raise ModelUnavailableError(self._realesrgan_status)

            try:
                enhanced = self._enhance_with_realesrgan(image, preset)
            except Exception as exc:  # pragma: no cover - optional ML runtime can fail in deployment-specific ways
                logger.exception("Real-ESRGAN inference failed")
                raise ModelRuntimeError("Real-ESRGAN failed during inference.") from exc
            engine_label = "RealESRGAN_x4plus"
        else:
            enhanced = self._fallback_enhance(image, preset)
            engine_label = "Pillow fallback"

        output = self._encode(enhanced)
        return EnhancementResult(
            image_bytes=output,
            content_type="image/jpeg",
            width=enhanced.width,
            height=enhanced.height,
            engine=engine,
            engine_label=engine_label,
            preset=preset,
        )

    def engine_status(self) -> dict[str, dict[str, object]]:
        return {
            "pillow_fallback": {
                "label": "Pillow fallback",
                "available": True,
                "detail": "Ready. Uses deterministic resizing, contrast, denoise, and sharpening.",
            },
            "realesrgan": {
                "label": "Real-ESRGAN x4plus",
                "available": self._realesrgan is not None,
                "detail": self._realesrgan_status,
                "model_path": str(REALESRGAN_MODEL_PATH),
            },
        }

    def _decode(self, file_bytes: bytes) -> Image.Image:
        if not file_bytes:
            raise ImageEnhancementError("Upload an image before enhancing.")

        try:
            image = Image.open(io.BytesIO(file_bytes))
            image = ImageOps.exif_transpose(image)
            image.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise ImageEnhancementError("The uploaded file is not a readable image.") from exc

        if image.width * image.height > self.max_input_pixels:
            raise ImageEnhancementError("Image is too large. Use a file under 24 megapixels.")

        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")

        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.getchannel("A"))
            image = background

        return image.convert("RGB")

    def _load_realesrgan(self):
        if not REALESRGAN_MODEL_PATH.exists():
            self._realesrgan_status = (
                f"Model weights not found at {REALESRGAN_MODEL_PATH}. "
                "Install the Real-ESRGAN extras and download RealESRGAN_x4plus.pth."
            )
            return None

        try:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except Exception as exc:
            self._realesrgan_status = (
                "Real-ESRGAN Python dependencies are not installed. "
                "Install torch, basicsr, realesrgan, and numpy."
            )
            logger.info("Real-ESRGAN imports are not ready: %s", exc)
            return None

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            upsampler = RealESRGANer(
                scale=4,
                model_path=str(REALESRGAN_MODEL_PATH),
                model=model,
                tile=256,
                tile_pad=10,
                pre_pad=0,
                half=device == "cuda",
                device=device,
            )
            self._realesrgan_status = f"Ready on {device} with {REALESRGAN_MODEL_PATH}."
            return upsampler
        except Exception as exc:
            self._realesrgan_status = f"Real-ESRGAN could not initialize: {exc}"
            logger.info("Real-ESRGAN is not ready: %s", exc)
            return None

    def _enhance_with_realesrgan(self, image: Image.Image, preset: PresetName) -> Image.Image:
        import numpy as np

        rgb = np.array(image)
        bgr = rgb[:, :, ::-1]
        output_bgr, _ = self._realesrgan.enhance(bgr, outscale=2)
        output_rgb = output_bgr[:, :, ::-1]
        enhanced = Image.fromarray(output_rgb)
        enhanced = self._fit_long_edge(enhanced, self.target_long_edge)
        return self._final_polish(enhanced, preset, denoise=False, strength="model")

    def _fallback_enhance(self, image: Image.Image, preset: PresetName) -> Image.Image:
        image = self._fit_long_edge(image, self.target_long_edge)
        return self._final_polish(image, preset, denoise=True, strength="fallback")

    def _final_polish(
        self,
        image: Image.Image,
        preset: PresetName,
        denoise: bool,
        strength: Literal["model", "fallback"],
    ) -> Image.Image:
        image = ImageOps.autocontrast(image, cutoff=1)

        if preset == "product_detail":
            color, contrast, sharpness = 1.04, 1.08, 1.26
            blur_radius, blend = 0.7, 0.28
        elif preset == "product_soft":
            color, contrast, sharpness = 1.01, 1.03, 1.08
            blur_radius, blend = 0.5, 0.18
        else:
            color, contrast, sharpness = 1.02, 1.05, 1.16
            blur_radius, blend = 0.6, 0.22

        if strength == "model":
            sharpness = max(1.02, sharpness - 0.06)
            blend *= 0.5

        if denoise:
            smoothed = image.filter(ImageFilter.MedianFilter(size=3))
            image = Image.blend(image, smoothed, blend)

        image = image.filter(ImageFilter.UnsharpMask(radius=blur_radius, percent=120, threshold=3))
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Color(image).enhance(color)
        image = ImageEnhance.Sharpness(image).enhance(sharpness)
        return image

    def _fit_long_edge(self, image: Image.Image, long_edge: int) -> Image.Image:
        current_long_edge = max(image.size)
        if current_long_edge == long_edge:
            return image

        scale = long_edge / current_long_edge
        new_size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def _encode(self, image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=94, optimize=True, progressive=True)
        return buffer.getvalue()
