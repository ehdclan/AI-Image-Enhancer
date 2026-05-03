from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RealESRGAN_x4plus weights to ONNX.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("weights/RealESRGAN_x4plus.pth"),
        help="Path to the RealESRGAN_x4plus .pth checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("weights/RealESRGAN_x4plus.onnx"),
        help="Destination ONNX path.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--height", type=int, default=64, help="Dummy input height for export.")
    parser.add_argument("--width", type=int, default=64, help="Dummy input width for export.")
    return parser.parse_args()


def load_rrdbnet():
    from basicsr.archs.rrdbnet_arch import RRDBNet

    return RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )


def resolve_state_dict(checkpoint) -> dict:
    if isinstance(checkpoint, dict):
        for key in ("params_ema", "params"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError("Unsupported checkpoint format.")


def apply_torch_onnx_compat() -> None:
    import torch.utils._pytree as pytree

    if hasattr(pytree, "register_pytree_node") or not hasattr(pytree, "_register_pytree_node"):
        return

    def compat_register_pytree_node(cls, flatten_fn, unflatten_fn, *args, **kwargs):
        return pytree._register_pytree_node(cls, flatten_fn, unflatten_fn)

    pytree.register_pytree_node = compat_register_pytree_node


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.input}")

    import torch

    apply_torch_onnx_compat()
    model = load_rrdbnet()
    checkpoint = torch.load(args.input, map_location="cpu")
    state_dict = resolve_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    dummy_input = torch.randn(1, 3, args.height, args.width, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(args.output),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height_x4", 3: "width_x4"},
        },
    )

    print(f"Exported ONNX model to: {args.output}")


if __name__ == "__main__":
    main()
