from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.enhancer import ProductImageEnhancer


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-evaluate studio_product on local store images.")
    parser.add_argument("input_dir", type=Path, help="Folder containing source product images.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/studio_product_eval"),
        help="Folder for enhanced images and the comparison sheet.",
    )
    parser.add_argument("--target-long-edge", type=int, default=1200)
    parser.add_argument(
        "--preset",
        choices=["product_standard", "product_detail", "product_soft"],
        default="product_standard",
    )
    return parser.parse_args()


def image_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def fit_thumbnail(image: Image.Image, height: int) -> Image.Image:
    scale = height / image.height
    size = (max(1, round(image.width * scale)), height)
    return image.resize(size, Image.Resampling.LANCZOS)


def build_comparison_row(
    source_path: Path,
    original: Image.Image,
    enhanced: Image.Image,
    mode: str,
    mask_label: str,
    height: int = 320,
) -> Image.Image:
    font = ImageFont.load_default()
    left = fit_thumbnail(original, height)
    right = fit_thumbnail(enhanced, height)
    row_width = left.width + right.width + 36
    row = Image.new("RGB", (row_width, height + 44), (246, 247, 248))
    row.paste(left, (0, 24))
    row.paste(right, (left.width + 36, 24))

    draw = ImageDraw.Draw(row)
    label = f"{source_path.name} | {mode} | {mask_label}"
    draw.text((4, 6), label, fill=(20, 24, 28), font=font)
    return row


def build_sheet(rows: list[Image.Image]) -> Image.Image:
    gap = 16
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows) + gap * (len(rows) - 1)
    sheet = Image.new("RGB", (width, height), (235, 237, 240))

    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height + gap

    return sheet


def main() -> None:
    args = parse_args()
    files = image_files(args.input_dir)
    if not files:
        raise SystemExit(f"No supported images found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    enhancer = ProductImageEnhancer(target_long_edge=args.target_long_edge)
    rows: list[Image.Image] = []

    for source_path in files:
        source_bytes = source_path.read_bytes()
        original = Image.open(io.BytesIO(source_bytes)).convert("RGB")
        decoded = enhancer._decode(source_bytes)
        mask = enhancer._segment_product_mask(decoded)
        stats = enhancer._mask_stats(mask, decoded.size) if mask is not None else None

        mode = "fallback"
        mask_label = "mask=none"
        if stats is not None:
            mode = "scene_crop" if enhancer._is_undersegmented_tall_product(stats) else "cutout"
            mask_label = (
                f"mask={stats.coverage:.1%} "
                f"w={stats.bbox_width_ratio:.0%} h={stats.bbox_height_ratio:.0%}"
            )

        result = enhancer.enhance(source_bytes, preset=args.preset, engine="studio_product")
        enhanced = Image.open(io.BytesIO(result.image_bytes)).convert("RGB")
        output_path = args.output_dir / f"{source_path.stem}-studio.jpg"
        enhanced.save(output_path, quality=92)

        rows.append(build_comparison_row(source_path, original, enhanced, mode, mask_label))
        print(f"{source_path.name}: {mode} -> {output_path}")

    sheet_path = args.output_dir / "comparison_sheet.jpg"
    build_sheet(rows).save(sheet_path, quality=90)
    print(f"comparison sheet: {sheet_path}")


if __name__ == "__main__":
    main()
