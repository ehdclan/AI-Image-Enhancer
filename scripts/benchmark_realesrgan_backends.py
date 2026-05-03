from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Real-ESRGAN ONNX vs PyTorch backends.")
    parser.add_argument("images", nargs="*", type=Path, help="Input images to benchmark.")
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of inference runs per backend and image. The first run is treated as cold/warmup.",
    )
    parser.add_argument(
        "--preset",
        default="product_standard",
        choices=["product_standard", "product_detail", "product_soft"],
        help="Preset to use for the enhancer call.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/realesrgan-benchmark"),
        help="Directory to store backend output images and summary JSON.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=["onnx", "pytorch"], help=argparse.SUPPRESS)
    parser.add_argument("--image", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def max_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def worker_mode(args: argparse.Namespace) -> int:
    os.environ["REALESRGAN_BACKEND"] = args.backend

    start = time.perf_counter()
    from app.enhancer import ProductImageEnhancer

    enhancer = ProductImageEnhancer()
    init_seconds = time.perf_counter() - start

    file_bytes = args.image.read_bytes()
    inference_seconds: list[float] = []
    result_payload: dict[str, object] = {}

    for index in range(args.runs):
        begin = time.perf_counter()
        result = enhancer.enhance(file_bytes, preset=args.preset, engine="realesrgan")
        inference_seconds.append(time.perf_counter() - begin)

        if index == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(result.image_bytes)
            result_payload = {
                "width": result.width,
                "height": result.height,
                "engine_label": result.engine_label,
            }

    payload = {
        "backend": args.backend,
        "image": str(args.image),
        "init_seconds": init_seconds,
        "first_inference_seconds": inference_seconds[0],
        "warm_inference_seconds": inference_seconds[1:],
        "warm_mean_seconds": statistics.mean(inference_seconds[1:]) if len(inference_seconds) > 1 else None,
        "warm_p95_seconds": percentile(inference_seconds[1:], 95) if len(inference_seconds) > 1 else None,
        "peak_rss_mb": max_rss_mb(),
        "output_path": str(args.output),
        **result_payload,
    }
    print(json.dumps(payload))
    return 0


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (pct / 100)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def run_worker(
    script_path: Path,
    backend: str,
    image_path: Path,
    output_path: Path,
    runs: int,
    preset: str,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(script_path),
        "--worker",
        "--backend",
        backend,
        "--image",
        str(image_path),
        "--output",
        str(output_path),
        "--runs",
        str(runs),
        "--preset",
        preset,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Worker failed for backend={backend} image={image_path.name}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Worker returned no JSON payload for backend={backend} image={image_path.name}")
    return json.loads(lines[-1])


def compare_outputs(path_a: Path, path_b: Path) -> dict[str, object]:
    import numpy as np
    from PIL import Image

    image_a = Image.open(path_a).convert("RGB")
    image_b = Image.open(path_b).convert("RGB")

    if image_a.size != image_b.size:
        return {
            "same_size": False,
            "size_a": image_a.size,
            "size_b": image_b.size,
        }

    array_a = np.asarray(image_a).astype("float32")
    array_b = np.asarray(image_b).astype("float32")
    diff = np.abs(array_a - array_b)
    mse = float(((array_a - array_b) ** 2).mean())
    psnr = float("inf") if mse == 0 else float(20 * math.log10(255.0) - 10 * math.log10(mse))
    return {
        "same_size": True,
        "size": image_a.size,
        "mae": float(diff.mean()),
        "max_abs_diff": int(diff.max()),
        "psnr_db": psnr,
    }


def default_images() -> list[Path]:
    base = Path("/Users/a1/Downloads/image_enhancer_v2/ALATPay Store Images")
    return [
        base / "1.jpg",
        base / "1719368575787-rggaqekbbdl.webp",
        base / "1741700996738-uekq3sds8wf.webp",
    ]


def main() -> int:
    args = parse_args()
    if args.worker:
        return worker_mode(args)

    images = args.images or default_images()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()

    summary: dict[str, object] = {"runs": args.runs, "preset": args.preset, "images": []}

    for image_path in images:
        image_path = image_path.resolve()
        onnx_output = output_dir / f"{image_path.stem}-onnx.jpg"
        pytorch_output = output_dir / f"{image_path.stem}-pytorch.jpg"

        onnx_metrics = run_worker(script_path, "onnx", image_path, onnx_output, args.runs, args.preset)
        pytorch_metrics = run_worker(script_path, "pytorch", image_path, pytorch_output, args.runs, args.preset)
        parity = compare_outputs(onnx_output, pytorch_output)

        image_summary = {
            "image": str(image_path),
            "onnx": onnx_metrics,
            "pytorch": pytorch_metrics,
            "parity": parity,
        }
        summary["images"].append(image_summary)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved benchmark summary to: {summary_path}")
    for image_summary in summary["images"]:
        image_name = Path(image_summary["image"]).name
        onnx_metrics = image_summary["onnx"]
        pytorch_metrics = image_summary["pytorch"]
        parity = image_summary["parity"]
        print(f"\n{image_name}")
        print(
            "  ONNX     "
            f"init={onnx_metrics['init_seconds']:.3f}s "
            f"first={onnx_metrics['first_inference_seconds']:.3f}s "
            f"warm_mean={_format_optional(onnx_metrics['warm_mean_seconds'])} "
            f"peak_rss={onnx_metrics['peak_rss_mb']:.1f}MB"
        )
        print(
            "  PyTorch  "
            f"init={pytorch_metrics['init_seconds']:.3f}s "
            f"first={pytorch_metrics['first_inference_seconds']:.3f}s "
            f"warm_mean={_format_optional(pytorch_metrics['warm_mean_seconds'])} "
            f"peak_rss={pytorch_metrics['peak_rss_mb']:.1f}MB"
        )
        if parity["same_size"]:
            print(
                "  Parity   "
                f"size={parity['size'][0]}x{parity['size'][1]} "
                f"mae={parity['mae']:.4f} "
                f"max_diff={parity['max_abs_diff']} "
                f"psnr={_format_psnr(parity['psnr_db'])}"
            )
        else:
            print(f"  Parity   size mismatch: {parity}")

    return 0


def _format_optional(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}s"


def _format_psnr(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.2f}dB"


if __name__ == "__main__":
    raise SystemExit(main())
