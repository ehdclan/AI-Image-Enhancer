from __future__ import annotations

import base64
import io
import importlib.util
import logging
import os
import warnings
from binascii import Error as BinasciiError
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


PresetName = Literal["product_standard", "product_detail", "product_soft"]
EngineName = Literal[
    "realesrgan",
    "pillow_fallback",
    "studio_product",
    "studio_product_realesrgan",
    "studio_product_focus",
    "studio_product_generative",
]

REALESRGAN_MODEL_PATH = Path(os.getenv("REALESRGAN_MODEL_PATH", "weights/RealESRGAN_x4plus.pth"))
REMBG_MODEL_DIR = Path(os.getenv("REMBG_MODEL_DIR", "weights/rembg"))
REMBG_MODEL_NAME = os.getenv("REMBG_MODEL_NAME", "isnet-general-use")
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2")
OPENAI_IMAGE_SIZE = os.getenv("OPENAI_IMAGE_SIZE", "1024x1024")
MAX_OUTPUT_LONG_EDGE = _env_int("MAX_OUTPUT_LONG_EDGE", 4096)
MAX_INPUT_BYTES = _env_int("MAX_INPUT_BYTES", 12_000_000)
ALLOWED_IMAGE_FORMATS = ("JPEG", "PNG", "WEBP")


@dataclass(frozen=True)
class EnhancementResult:
    image_bytes: bytes
    content_type: str
    width: int
    height: int
    engine: EngineName
    engine_label: str
    preset: PresetName


@dataclass(frozen=True)
class MaskStats:
    coverage: float
    bbox: tuple[int, int, int, int]
    bbox_width_ratio: float
    bbox_height_ratio: float


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

    def __init__(
        self,
        max_input_bytes: int = MAX_INPUT_BYTES,
        max_input_pixels: int = 24_000_000,
        target_long_edge: int = 1800,
        max_output_long_edge: int = MAX_OUTPUT_LONG_EDGE,
    ) -> None:
        self.max_input_bytes = max(1, max_input_bytes)
        self.max_input_pixels = max_input_pixels
        self.target_long_edge = max(1, target_long_edge)
        self.max_output_long_edge = max(self.target_long_edge, max_output_long_edge)
        self.allowed_formats = ALLOWED_IMAGE_FORMATS
        self._realesrgan_status = "Checking Real-ESRGAN runtime."
        self._realesrgan = self._load_realesrgan()
        self._rembg_session = None
        self._rembg_session_model: Optional[str] = None

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
        elif engine == "studio_product_realesrgan":
            if self._realesrgan is None:
                self._realesrgan = self._load_realesrgan()

            if self._realesrgan is None:
                raise ModelUnavailableError(self._realesrgan_status)

            try:
                enhanced = self._studio_product_realesrgan_enhance(image, preset)
            except Exception as exc:  # pragma: no cover - optional ML runtime can fail in deployment-specific ways
                logger.exception("Studio product focus + Real-ESRGAN inference failed")
                raise ModelRuntimeError("Studio product focus + Real-ESRGAN failed during inference.") from exc
            engine_label = "Studio product focus + RealESRGAN_x4plus"
        elif engine == "studio_product_focus":
            enhanced = self._studio_product_focus_enhance(image, preset)
            engine_label = "Studio product focus"
        elif engine == "studio_product_generative":
            enhanced = self._studio_product_generative_enhance(image, preset)
            engine_label = f"Studio product generative ({OPENAI_IMAGE_MODEL})"
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
                "detail": self._studio_engine_detail(),
            },
            "studio_product_realesrgan": {
                "label": "Studio product focus + Real-ESRGAN",
                "available": self._realesrgan is not None,
                "detail": self._studio_product_realesrgan_detail(),
            },
            "studio_product_focus": {
                "label": "Studio product focus",
                "available": True,
                "detail": self._focus_engine_detail(),
            },
            "studio_product_generative": {
                "label": "Studio product generative",
                "available": self._is_generative_ready(),
                "detail": self._generative_engine_detail(),
            },
            "realesrgan": {
                "label": "Real-ESRGAN x4plus",
                "available": self._realesrgan is not None,
                "detail": self._realesrgan_status,
            },
        }

    def _decode(self, file_bytes: bytes) -> Image.Image:
        if not file_bytes:
            raise ImageEnhancementError("Upload an image before enhancing.")

        if len(file_bytes) > self.max_input_bytes:
            raise ImageEnhancementError(
                f"Image is too large. Use a file under {self.max_input_bytes // 1_000_000} MB."
            )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                image = Image.open(io.BytesIO(file_bytes), formats=self.allowed_formats)
                if image.width * image.height > self.max_input_pixels:
                    raise ImageEnhancementError("Image is too large. Use a file under 24 megapixels.")

                image = ImageOps.exif_transpose(image)
                if image.width * image.height > self.max_input_pixels:
                    raise ImageEnhancementError("Image is too large. Use a file under 24 megapixels.")

                image.load()
        except Image.DecompressionBombError as exc:
            raise ImageEnhancementError("Image is too large. Use a file under 24 megapixels.") from exc
        except Image.DecompressionBombWarning as exc:
            raise ImageEnhancementError("Image is too large. Use a file under 24 megapixels.") from exc
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ImageEnhancementError("Upload a readable JPEG, PNG, or WebP image.") from exc

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
                "Model weights not found. Install the Real-ESRGAN extras and add RealESRGAN_x4plus.pth."
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
            self._realesrgan_status = f"Ready on {device}."
            return upsampler
        except Exception as exc:
            self._realesrgan_status = "Real-ESRGAN could not initialize."
            logger.info("Real-ESRGAN is not ready: %s", exc)
            return None

    def _enhance_with_realesrgan(self, image: Image.Image, preset: PresetName) -> Image.Image:
        import numpy as np

        rgb = np.array(image)
        bgr = rgb[:, :, ::-1]
        output_bgr, _ = self._realesrgan.enhance(bgr, outscale=2)
        output_rgb = output_bgr[:, :, ::-1]
        enhanced = Image.fromarray(output_rgb)
        enhanced = self._fit_long_edge(enhanced, self._output_long_edge_for(image))
        return self._final_polish(enhanced, preset, denoise=False, strength="model")

    def _fallback_enhance(self, image: Image.Image, preset: PresetName) -> Image.Image:
        image = self._fit_long_edge(image, self._output_long_edge_for(image))
        return self._final_polish(image, preset, denoise=True, strength="fallback")

    def _studio_product_enhance(self, image: Image.Image, preset: PresetName) -> Image.Image:
        mask = self._segment_product_mask(image)
        if mask is None:
            return self._fallback_enhance(image, preset)

        stats = self._mask_stats(mask, image.size)
        if stats is None:
            return self._fallback_enhance(image, preset)

        bbox = self._expand_box(stats.bbox, image.size, padding=max(4, round(max(image.size) * 0.015)))
        product = image.crop(bbox)
        product_mask = mask.crop(bbox)
        product = self._final_polish(product, preset, denoise=True, strength="studio")

        product_rgba = product.convert("RGBA")
        product_rgba.putalpha(product_mask)

        canvas_size = self._output_long_edge_for(image)
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

    def _studio_scene_crop_enhance(
        self,
        image: Image.Image,
        bbox: tuple[int, int, int, int],
        preset: PresetName,
    ) -> Image.Image:
        crop_box = self._expand_scene_box(bbox, image.size)
        crop = image.crop(crop_box)
        crop = self._final_polish(crop, preset, denoise=True, strength="studio")

        canvas_size = self._output_long_edge_for(image)
        background = ImageOps.fit(crop, (canvas_size, canvas_size), method=Image.Resampling.LANCZOS)
        background = background.filter(ImageFilter.GaussianBlur(radius=max(14, canvas_size // 55)))
        background = ImageEnhance.Color(background).enhance(0.25)
        background = ImageEnhance.Brightness(background).enhance(1.15)
        canvas = Image.blend(background, self._studio_background(canvas_size), 0.78)

        max_crop_size = round(canvas_size * 0.9)
        crop = ImageOps.contain(crop, (max_crop_size, max_crop_size), method=Image.Resampling.LANCZOS)
        crop_x = (canvas_size - crop.width) // 2
        crop_y = (canvas_size - crop.height) // 2
        crop_mask = self._soft_scene_crop_mask(crop.size)
        canvas.paste(crop, (crop_x, crop_y), crop_mask)
        return canvas.convert("RGB")

    def _studio_product_realesrgan_enhance(self, image: Image.Image, preset: PresetName) -> Image.Image:
        studio_image = self._studio_product_focus_enhance(image, preset)
        return self._enhance_with_realesrgan(studio_image, preset)

    def _studio_product_focus_enhance(self, image: Image.Image, preset: PresetName) -> Image.Image:
        mask = self._segment_product_mask(image)
        if mask is None:
            return self._fallback_enhance(image, preset)

        focus_mask = self._product_focus_mask(mask, image.size)
        product_scene = self._final_polish(image, preset, denoise=True, strength="studio")
        focused_scene = Image.composite(product_scene, image, focus_mask)
        return self._fit_long_edge(focused_scene.convert("RGB"), self._output_long_edge_for(image))

    def _studio_product_generative_enhance(self, image: Image.Image, preset: PresetName) -> Image.Image:
        if not os.getenv("OPENAI_API_KEY"):
            raise ModelUnavailableError("Set OPENAI_API_KEY to use studio_product_generative.")

        if importlib.util.find_spec("openai") is None:
            raise ModelUnavailableError("Install the openai Python package to use studio_product_generative.")

        reference = self._studio_product_enhance(image, preset)

        try:
            generated = self._enhance_with_openai_image_edit(reference)
        except ModelRuntimeError:
            raise
        except Exception as exc:  # pragma: no cover - external API behavior varies by account/model
            logger.exception("OpenAI image edit failed")
            raise ModelRuntimeError("Generative studio enhancement failed during OpenAI image editing.") from exc

        generated = generated.convert("RGB")
        generated = self._fit_long_edge(generated, self._output_long_edge_for(image))
        return self._final_polish(generated, preset, denoise=False, strength="model")

    def _studio_engine_detail(self) -> str:
        if importlib.util.find_spec("rembg") is not None:
            return (
                f"Ready with rembg {REMBG_MODEL_NAME} segmentation, studio canvas, "
                "soft shadow, and safe polish."
            )

        if importlib.util.find_spec("cv2") is not None:
            return "Ready with OpenCV GrabCut segmentation, studio canvas, soft shadow, and safe polish."

        return "Ready with basic masking only. Install opencv-python-headless or rembg for stronger product cutouts."

    def _focus_engine_detail(self) -> str:
        if importlib.util.find_spec("rembg") is not None:
            return (
                f"Ready with rembg {REMBG_MODEL_NAME} focus masking. Keeps the original background "
                "and enhances only the detected product area."
            )

        if importlib.util.find_spec("cv2") is not None:
            return "Ready with OpenCV focus masking. Keeps the original background and enhances the product area."

        return "Ready with full-scene fallback polish. Install rembg for product-only focus."

    def _studio_product_realesrgan_detail(self) -> str:
        if self._realesrgan is None:
            return f"Not ready. {self._realesrgan_status}"

        return (
            "Ready. Runs studio_product_focus first to enhance the product while keeping the original scene, "
            "then applies Real-ESRGAN restoration for rough or low-quality source images."
        )

    def _is_generative_ready(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY")) and importlib.util.find_spec("openai") is not None

    def _generative_engine_detail(self) -> str:
        if importlib.util.find_spec("openai") is None:
            return "Not ready. Install the openai Python package."

        if not os.getenv("OPENAI_API_KEY"):
            return "Not ready. Set OPENAI_API_KEY to enable the controlled generative studio-photo pass."

        return (
            "Ready with a constrained generative edit pass after studio_product to preserve the product while "
            "improving lighting, background, and camera finish."
        )

    def _enhance_with_openai_image_edit(self, reference: Image.Image) -> Image.Image:
        from openai import OpenAI

        client = OpenAI()
        reference_file = io.BytesIO(self._image_to_png_bytes(reference))
        reference_file.name = "product-reference.png"

        result = client.images.edit(
            model=OPENAI_IMAGE_MODEL,
            image=reference_file,
            prompt=self._generative_product_prompt(),
            size=OPENAI_IMAGE_SIZE,
        )

        if not result.data or not result.data[0].b64_json:
            raise ModelRuntimeError("OpenAI image edit did not return image bytes.")

        try:
            output_bytes = base64.b64decode(result.data[0].b64_json)
            output = Image.open(io.BytesIO(output_bytes))
            output.load()
            return output
        except (BinasciiError, ValueError, OSError) as exc:
            raise ModelRuntimeError("OpenAI image edit returned unreadable image bytes.") from exc

    def _generative_product_prompt(self) -> str:
        return (
            "Create a premium ecommerce product photo from this reference image. "
            "Preserve the exact product identity, shape, proportions, colors, material texture, logos, labels, "
            "printed text, packaging edges, and item count. Do not invent, replace, rewrite, remove, or stylize "
            "any branding or product text. Improve only the catalog photography: clean off-white studio background, "
            "balanced DSLR or mirrorless camera lighting, natural soft contact shadow, crisp focus, realistic color, "
            "and subtle high-end product-photo finish. Keep the product centered and fully visible. "
            "No hands, props, extra objects, decorative elements, watermarks, fake labels, or lifestyle scene."
        )

    def _image_to_png_bytes(self, image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        buffer.seek(0)
        return buffer.getvalue()

    def _segment_product_mask(self, image: Image.Image) -> Optional[Image.Image]:
        mask = self._build_rembg_product_mask(image)
        if mask is not None:
            return mask

        mask = self._build_grabcut_product_mask(image)
        if mask is not None:
            return mask

        return self._build_heuristic_product_mask(image)

    def _build_rembg_product_mask(self, image: Image.Image) -> Optional[Image.Image]:
        try:
            from rembg import new_session, remove
        except Exception:
            return None

        try:
            REMBG_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("U2NET_HOME", str(REMBG_MODEL_DIR.resolve()))

            if self._rembg_session is None or self._rembg_session_model != REMBG_MODEL_NAME:
                self._rembg_session = new_session(REMBG_MODEL_NAME, providers=["CPUExecutionProvider"])
                self._rembg_session_model = REMBG_MODEL_NAME

            output = remove(image, session=self._rembg_session, post_process_mask=True)
            if output.mode != "RGBA":
                output = output.convert("RGBA")

            mask = self._clean_product_mask(output.getchannel("A"), image.size)
            return self._validate_product_mask(mask, image.size, allow_border_touch=True)
        except Exception as exc:
            logger.info("rembg segmentation failed; falling back: %s", exc)
            return None

    def _build_grabcut_product_mask(self, image: Image.Image) -> Optional[Image.Image]:
        try:
            import cv2
            import numpy as np
        except Exception:
            return None

        try:
            max_side = 900
            scale = min(1.0, max_side / max(image.size))
            work = image
            if scale < 1.0:
                work = image.resize(
                    (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                    Image.Resampling.LANCZOS,
                )

            rgb = np.array(work)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            height, width = bgr.shape[:2]
            gc_mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)

            border_x = max(3, round(width * 0.035))
            border_y = max(3, round(height * 0.035))
            gc_mask[:border_y, :] = cv2.GC_BGD
            gc_mask[height - border_y :, :] = cv2.GC_BGD
            gc_mask[:, :border_x] = cv2.GC_BGD
            gc_mask[:, width - border_x :] = cv2.GC_BGD

            heuristic = self._build_heuristic_product_mask(work)
            if heuristic is not None:
                heuristic_array = np.array(heuristic)
                probable_foreground = heuristic_array > 48
                strong_foreground = heuristic_array > 220
                gc_mask[probable_foreground] = cv2.GC_PR_FGD
                gc_mask[strong_foreground] = cv2.GC_FGD
                mode = cv2.GC_INIT_WITH_MASK
                rect = (0, 0, 1, 1)
            else:
                rect = (
                    max(1, round(width * 0.05)),
                    max(1, round(height * 0.05)),
                    max(2, round(width * 0.9)),
                    max(2, round(height * 0.9)),
                )
                mode = cv2.GC_INIT_WITH_RECT

            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            cv2.grabCut(bgr, gc_mask, rect, bgd_model, fgd_model, 5, mode)

            foreground = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                255,
                0,
            ).astype("uint8")

            kernel_size = max(3, round(min(width, height) * 0.01))
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)
            foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel, iterations=1)

            mask = Image.fromarray(foreground)
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.Resampling.LANCZOS)
            mask = self._clean_product_mask(mask, image.size)
            return self._validate_product_mask(mask, image.size, allow_border_touch=False)
        except Exception as exc:
            logger.info("OpenCV GrabCut segmentation failed; falling back: %s", exc)
            return None

    def _build_heuristic_product_mask(self, image: Image.Image) -> Optional[Image.Image]:
        background = Image.new("RGB", image.size, self._estimate_background_color(image))
        diff = ImageChops.difference(image, background).convert("L")
        stats = ImageStat.Stat(diff)
        threshold = max(18, min(58, round(stats.mean[0] + stats.stddev[0] * 0.75)))
        mask = diff.point(lambda pixel: 255 if pixel > threshold else 0, mode="L")
        mask = mask.filter(ImageFilter.MedianFilter(size=5))
        mask = mask.filter(ImageFilter.MaxFilter(size=9))
        mask = mask.filter(ImageFilter.MinFilter(size=5))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=1.1))

        return self._validate_product_mask(mask, image.size, allow_border_touch=False)

    def _clean_product_mask(
        self,
        mask: Image.Image,
        image_size: tuple[int, int],
    ) -> Image.Image:
        try:
            import cv2
            import numpy as np
        except Exception:
            return mask.convert("L").filter(ImageFilter.GaussianBlur(radius=0.8))

        mask_array = np.array(mask.convert("L"))
        hard_mask = (mask_array > 32).astype("uint8")
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(hard_mask, 8)

        if component_count <= 1:
            return Image.fromarray((hard_mask * 255).astype("uint8"))

        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_area = int(areas.max()) if len(areas) else 0
        min_area = max(round(image_size[0] * image_size[1] * 0.003), round(largest_area * 0.08), 24)
        cleaned = np.zeros_like(hard_mask)

        for label in range(1, component_count):
            left = stats[label, cv2.CC_STAT_LEFT]
            top = stats[label, cv2.CC_STAT_TOP]
            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]
            right = left + width
            bottom = top + height

            touches_border = left <= 1 or top <= 1 or right >= image_size[0] - 1 or bottom >= image_size[1] - 1
            if area < min_area:
                continue
            if touches_border and area < largest_area * 0.65:
                continue

            cleaned[labels == label] = 255

        if not cleaned.any():
            cleaned[labels == int(areas.argmax()) + 1] = 255

        kernel_size = max(3, round(min(image_size) * 0.004))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

        cleaned_mask = Image.fromarray(cleaned.astype("uint8"))
        return cleaned_mask.filter(ImageFilter.GaussianBlur(radius=0.9))

    def _validate_product_mask(
        self,
        mask: Image.Image,
        image_size: tuple[int, int],
        allow_border_touch: bool,
    ) -> Optional[Image.Image]:
        mask = mask.convert("L").filter(ImageFilter.GaussianBlur(radius=0.6))
        histogram = mask.histogram()
        total_pixels = image_size[0] * image_size[1]
        coverage = sum(count for value, count in enumerate(histogram) if value > 24) / total_pixels

        if coverage < 0.02 or coverage > 0.88:
            return None

        hard_mask = mask.point(lambda pixel: 255 if pixel > 24 else 0, mode="L")
        bbox = hard_mask.getbbox()
        if bbox is None:
            return None

        box_width = bbox[2] - bbox[0]
        box_height = bbox[3] - bbox[1]
        if box_width >= image_size[0] * 0.98 and box_height >= image_size[1] * 0.98:
            return None

        bbox_area_ratio = (box_width * box_height) / total_pixels
        if not allow_border_touch and box_width > image_size[0] * 0.75 and box_height > image_size[1] * 0.75:
            if coverage < 0.45 and bbox_area_ratio > 0.55:
                return None

        if not allow_border_touch:
            border = max(2, round(min(image_size) * 0.025))
            top_band = hard_mask.crop((0, 0, image_size[0], border))
            bottom_band = hard_mask.crop((0, image_size[1] - border, image_size[0], image_size[1]))
            left_band = hard_mask.crop((0, 0, border, image_size[1]))
            right_band = hard_mask.crop((image_size[0] - border, 0, image_size[0], image_size[1]))
            border_pixels = sum(1 for band in (top_band, bottom_band, left_band, right_band) for pixel in band.getdata() if pixel)
            border_area = (top_band.width * top_band.height) + (bottom_band.width * bottom_band.height)
            border_area += (left_band.width * left_band.height) + (right_band.width * right_band.height)
            border_coverage = border_pixels / max(1, border_area)
            edge_hits = sum(
                [
                    bbox[0] <= border,
                    bbox[1] <= border,
                    bbox[2] >= image_size[0] - border,
                    bbox[3] >= image_size[1] - border,
                ]
            )

            if edge_hits >= 2 and (coverage > 0.18 or border_coverage > 0.08):
                return None

        return mask

    def _mask_stats(self, mask: Image.Image, image_size: tuple[int, int]) -> Optional[MaskStats]:
        histogram = mask.histogram()
        total_pixels = image_size[0] * image_size[1]
        coverage = sum(count for value, count in enumerate(histogram) if value > 24) / total_pixels
        hard_mask = mask.point(lambda pixel: 255 if pixel > 24 else 0, mode="L")
        bbox = hard_mask.getbbox()
        if bbox is None:
            return None

        box_width = bbox[2] - bbox[0]
        box_height = bbox[3] - bbox[1]
        return MaskStats(
            coverage=coverage,
            bbox=bbox,
            bbox_width_ratio=box_width / image_size[0],
            bbox_height_ratio=box_height / image_size[1],
        )

    def _is_undersegmented_tall_product(self, stats: MaskStats) -> bool:
        return (
            stats.bbox_height_ratio >= 0.78
            and stats.bbox_width_ratio <= 0.42
            and stats.coverage <= 0.22
        )

    def _has_top_contaminated_mask(self, stats: MaskStats, image_size: tuple[int, int]) -> bool:
        top_margin = max(4, round(image_size[1] * 0.015))
        return (
            stats.bbox[1] <= top_margin
            and stats.bbox_width_ratio >= 0.58
            and stats.coverage <= 0.3
        )

    def _should_use_scene_crop(self, stats: MaskStats, image_size: tuple[int, int]) -> bool:
        return self._is_undersegmented_tall_product(stats) or self._has_top_contaminated_mask(stats, image_size)

    def _product_focus_mask(self, mask: Image.Image, image_size: tuple[int, int]) -> Image.Image:
        focus = mask.convert("L").point(lambda pixel: 255 if pixel > 24 else 0, mode="L")
        stats = self._mask_stats(mask, image_size)

        if stats is not None and self._should_use_scene_crop(stats, image_size):
            repair = Image.new("L", image_size, 0)
            ImageDraw.Draw(repair).rectangle(self._expand_scene_box(stats.bbox, image_size), fill=255)
            focus = ImageChops.lighter(focus, repair)

        expand_size = self._odd_filter_size(round(min(image_size) * 0.02), minimum=9)
        focus = focus.filter(ImageFilter.MaxFilter(expand_size))
        blur_radius = max(5, round(min(image_size) * 0.01))
        return focus.filter(ImageFilter.GaussianBlur(radius=blur_radius))

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

    def _expand_scene_box(
        self,
        box: tuple[int, int, int, int],
        image_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        left, top, right, bottom = box
        width, height = image_size
        box_width = right - left
        box_height = bottom - top
        crop_width = min(width, max(round(box_width * 1.22), round(width * 0.42), 12))
        crop_height = min(height, max(round(box_height * 1.12), round(height * 0.55), 12))
        center_x = (left + right) / 2
        center_y = top + box_height * 0.52

        crop_left = round(center_x - crop_width / 2)
        crop_top = round(center_y - crop_height / 2)
        crop_left = max(0, min(width - crop_width, crop_left))
        crop_top = max(0, min(height - crop_height, crop_top))

        return (
            crop_left,
            crop_top,
            crop_left + crop_width,
            crop_top + crop_height,
        )

    def _odd_filter_size(self, value: int, minimum: int) -> int:
        size = max(minimum, value)
        if size % 2 == 0:
            size += 1
        return size

    def _fit_product_on_canvas(self, product: Image.Image, canvas_size: int) -> Image.Image:
        max_width = round(canvas_size * 0.86)
        max_height = round(canvas_size * 0.82)
        scale = min(max_width / product.width, max_height / product.height)
        new_size = (
            max(1, round(product.width * scale)),
            max(1, round(product.height * scale)),
        )
        return product.resize(new_size, Image.Resampling.LANCZOS)

    def _soft_scene_crop_mask(self, size: tuple[int, int]) -> Image.Image:
        width, height = size
        mask = Image.new("L", size, 0)
        inset = max(6, round(min(size) * 0.025))
        radius = max(10, round(min(size) * 0.035))
        ImageDraw.Draw(mask).rounded_rectangle(
            (inset, inset, width - inset, height - inset),
            radius=radius,
            fill=255,
        )
        return mask.filter(ImageFilter.GaussianBlur(radius=max(6, round(min(size) * 0.018))))

    def _output_long_edge_for(self, image: Image.Image) -> int:
        desired_long_edge = max(self.target_long_edge, max(image.size))
        return min(desired_long_edge, self.max_output_long_edge)

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
        shadow_width = max(1, round(alpha.width * 0.9))
        shadow_height = max(18, round(canvas_size * 0.055))
        shadow = Image.new("L", (shadow_width, shadow_height), 0)
        draw = ImageDraw.Draw(shadow)
        inset_x = max(1, round(shadow_width * 0.08))
        inset_y = max(1, round(shadow_height * 0.22))
        draw.ellipse((inset_x, inset_y, shadow_width - inset_x, shadow_height - inset_y), fill=255)
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(10, canvas_size // 70)))
        shadow = shadow.point(lambda pixel: round(pixel * 0.20))

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
