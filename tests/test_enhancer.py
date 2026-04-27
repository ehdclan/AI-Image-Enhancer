from __future__ import annotations

import unittest

from PIL import Image

from app.enhancer import ProductImageEnhancer


class _TrackingEnhancer(ProductImageEnhancer):
    def __init__(self) -> None:
        super().__init__()
        self.scene_crop_used = False

    def _segment_product_mask(self, image: Image.Image) -> Image.Image:
        mask = Image.new("L", image.size, 0)
        left = round(image.width * 0.39)
        top = round(image.height * 0.04)
        right = round(image.width * 0.79)
        bottom = round(image.height * 0.94)
        mask.paste(255, (left, top, right, bottom))
        return mask

    def _studio_scene_crop_enhance(
        self,
        image: Image.Image,
        bbox: tuple[int, int, int, int],
        preset: str,
    ) -> Image.Image:
        self.scene_crop_used = True
        return image


class StudioProductRegressionTests(unittest.TestCase):
    def test_studio_product_keeps_cutout_path_for_tall_masks(self) -> None:
        enhancer = _TrackingEnhancer()
        image = Image.new("RGB", (1800, 1800), (255, 255, 255))
        result = enhancer._studio_product_enhance(image, "product_standard")

        self.assertFalse(enhancer.scene_crop_used)
        self.assertEqual(result.size, (1800, 1800))

    def test_studio_product_realesrgan_runs_studio_then_restoration(self) -> None:
        class HybridTrackingEnhancer(ProductImageEnhancer):
            def __init__(self) -> None:
                super().__init__()
                self._realesrgan = object()
                self.calls: list[str] = []

            def _decode(self, file_bytes: bytes) -> Image.Image:
                self.calls.append("decode")
                return Image.new("RGB", (64, 64), (255, 255, 255))

            def _studio_product_focus_enhance(self, image: Image.Image, preset: str) -> Image.Image:
                self.calls.append("studio_product_focus")
                return image

            def _enhance_with_realesrgan(self, image: Image.Image, preset: str) -> Image.Image:
                self.calls.append("realesrgan")
                return image

            def _encode(self, image: Image.Image) -> bytes:
                self.calls.append("encode")
                return b"jpeg"

        enhancer = HybridTrackingEnhancer()
        result = enhancer.enhance(b"raw", preset="product_standard", engine="studio_product_realesrgan")

        self.assertEqual(enhancer.calls, ["decode", "studio_product_focus", "realesrgan", "encode"])
        self.assertEqual(result.engine, "studio_product_realesrgan")

    def test_ultra_upscale_runs_restore_pipeline(self) -> None:
        class UltraTrackingEnhancer(ProductImageEnhancer):
            def __init__(self) -> None:
                super().__init__()
                self._realesrgan = object()
                self.calls: list[str] = []

            def _decode(self, file_bytes: bytes) -> Image.Image:
                self.calls.append("decode")
                return Image.new("RGB", (128, 96), (245, 244, 242))

            def _ultra_upscale_profile(self, image: Image.Image) -> str:
                self.calls.append("profile")
                return "product"

            def _prepare_ultra_upscale_source(self, image: Image.Image, profile: str) -> Image.Image:
                self.calls.append(f"prepare:{profile}")
                return image

            def _ultra_upscale_target_long_edge(self, image: Image.Image, profile: str) -> int:
                self.calls.append(f"target:{profile}")
                return 512

            def _run_realesrgan_raw(self, image: Image.Image, outscale: int) -> Image.Image:
                self.calls.append(f"realesrgan:{outscale}")
                return Image.new("RGB", (512, 384), (255, 255, 255))

            def _ultra_upscale_refine(
                self,
                restored: Image.Image,
                baseline: Image.Image,
                preset: str,
                profile: str,
            ) -> Image.Image:
                self.calls.append(f"refine:{profile}:{restored.size}:{baseline.size}")
                return restored

            def _encode(self, image: Image.Image) -> bytes:
                self.calls.append("encode")
                return b"jpeg"

        enhancer = UltraTrackingEnhancer()
        result = enhancer.enhance(b"raw", preset="product_standard", engine="ultra_upscale")

        self.assertEqual(
            enhancer.calls,
            [
                "decode",
                "profile",
                "prepare:product",
                "target:product",
                "realesrgan:4",
                "refine:product:(512, 384):(512, 384)",
                "encode",
            ],
        )
        self.assertEqual(result.engine, "ultra_upscale")


if __name__ == "__main__":
    unittest.main()
