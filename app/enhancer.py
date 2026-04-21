from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError

logger = logging.getLogger(__name__)

PresetName = Literal["product_standard", "product_detail", "product_soft"]
EngineName = Literal["realesrgan", "pillow_fallback", "studio_product"]

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
        elif engine == "studio_product":
            enhanced = self._studio_product_enhance(image, preset)
            engine_label = "Studio product"
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
            "studio_product": {
                "label": "Studio product",
                "available": True,
                "detail": "Ready. Cleans simple backgrounds, centers the product, adds studio shadow, and applies safe polish.",
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

    def _studio_product_enhance(self, image: Image.Image, preset: PresetName) -> Image.Image:
        mask = self._build_product_mask(image)
        bbox = mask.getbbox()

        if bbox is None:
            return self._fallback_enhance(image, preset)

        bbox = self._expand_box(bbox, image.size, padding=max(4, round(max(image.size) * 0.015)))
        product = image.crop(bbox)
        product_mask = mask.crop(bbox)
        product = self._final_polish(product, preset, denoise=True, strength="studio")

        product_rgba = product.convert("RGBA")
        product_rgba.putalpha(product_mask)

        canvas_size = self.target_long_edge
        product_rgba = self._fit_product_on_canvas(product_rgba, canvas_size)
        product_alpha = product_rgba.getchannel("A")

        canvas = self._studio_background(canvas_size)
        product_x = (canvas_size - product_rgba.width) // 2
        product_y = self._studio_product_y(canvas_size, product_rgba.height)

        shadow = self._build_shadow(product_alpha, canvas_size)
        shadow_x = (canvas_size - shadow.width) // 2
        shadow_y = min(
            canvas_size - shadow.height - max(8, canvas_size // 90),
            product_y + product_rgba.height - shadow.height // 2,
        )

        canvas.paste(shadow, (shadow_x, shadow_y), shadow)
        canvas.paste(product_rgba, (product_x, product_y), product_alpha)
        return canvas.convert("RGB")

    def _build_product_mask(self, image: Image.Image) -> Image.Image:
        background = Image.new("RGB", image.size, self._estimate_background_color(image))
        diff = ImageChops.difference(image, background).convert("L")
        stats = ImageStat.Stat(diff)
        threshold = max(18, min(58, round(stats.mean[0] + stats.stddev[0] * 0.75)))
        mask = diff.point(lambda pixel: 255 if pixel > threshold else 0, mode="L")
        mask = mask.filter(ImageFilter.MedianFilter(size=5))
        mask = mask.filter(ImageFilter.MaxFilter(size=9))
        mask = mask.filter(ImageFilter.MinFilter(size=5))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=1.1))

        histogram = mask.histogram()
        coverage = sum(count for value, count in enumerate(histogram) if value > 8) / (image.width * image.height)
        if coverage < 0.015:
            return Image.new("L", image.size, 255)

        if coverage > 0.92:
            return Image.new("L", image.size, 255)

        return mask

    def _estimate_background_color(self, image: Image.Image) -> tuple[int, int, int]:
        width, height = image.size
        strip = max(1, round(min(width, height) * 0.04))
        regions = [
            image.crop((0, 0, width, strip)),
            image.crop((0, height - strip, width, height)),
            image.crop((0, 0, strip, height)),
            image.crop((width - strip, 0, width, height)),
        ]
        colors = [region.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0)) for region in regions]
        return tuple(round(sum(color[channel] for color in colors) / len(colors)) for channel in range(3))

    def _expand_box(
        self,
        box: tuple[int, int, int, int],
        image_size: tuple[int, int],
        padding: int,
    ) -> tuple[int, int, int, int]:
        left, top, right, bottom = box
        width, height = image_size
        return (
            max(0, left - padding),
            max(0, top - padding),
            min(width, right + padding),
            min(height, bottom + padding),
        )

    def _fit_product_on_canvas(self, product: Image.Image, canvas_size: int) -> Image.Image:
        max_width = round(canvas_size * 0.78)
        max_height = round(canvas_size * 0.76)
        scale = min(max_width / product.width, max_height / product.height)
        new_size = (
            max(1, round(product.width * scale)),
            max(1, round(product.height * scale)),
        )
        return product.resize(new_size, Image.Resampling.LANCZOS)

    def _studio_product_y(self, canvas_size: int, product_height: int) -> int:
        top_limit = round(canvas_size * 0.08)
        bottom_limit = canvas_size - product_height - round(canvas_size * 0.13)
        centered = round((canvas_size - product_height) * 0.46)
        return max(top_limit, min(centered, bottom_limit))

    def _studio_background(self, size: int) -> Image.Image:
        top = (250, 251, 249)
        bottom = (239, 242, 240)
        strip = Image.new("RGB", (1, size), top)
        pixels = strip.load()

        for y in range(size):
            ratio = y / max(1, size - 1)
            color = tuple(round(top[channel] * (1 - ratio) + bottom[channel] * ratio) for channel in range(3))
            pixels[0, y] = color

        return strip.resize((size, size), Image.Resampling.BOX)

    def _build_shadow(self, alpha: Image.Image, canvas_size: int) -> Image.Image:
        shadow_height = max(1, round(alpha.height * 0.18))
        shadow = alpha.resize((alpha.width, shadow_height), Image.Resampling.LANCZOS)
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(12, canvas_size // 45)))
        shadow = shadow.point(lambda pixel: round(pixel * 0.24))

        shadow_layer = Image.new("RGBA", shadow.size, (18, 22, 20, 0))
        shadow_layer.putalpha(shadow)
        return shadow_layer

    def _final_polish(
        self,
        image: Image.Image,
        preset: PresetName,
        denoise: bool,
        strength: Literal["model", "fallback", "studio"],
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
