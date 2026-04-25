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


if __name__ == "__main__":
    unittest.main()
