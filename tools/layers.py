#!/usr/bin/env python3
"""Suggest an ncnn-zig `.layers` allowlist from an ncnn `.param` file.

The suggestion is meant to run, not to be minimal: it includes the wrapper
helper layers and fused activation layers. Remove layers you do not need and
test the model. The ncnn-zig build still expands static dependencies such as
Convolution -> Gemm automatically.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT_HELPERS = (
    "Packing",
    "Flatten",
    "Padding",
    "Interp",
    "Crop",
    "Cast",
    "Bias",
    "Scale",
)

FUSED_OWNERS = {
    "Convolution",
    "ConvolutionDepthWise",
    "Deconvolution",
    "DeconvolutionDepthWise",
    "DeformableConv2D",
}

ACTIVATIONS = {
    1: "ReLU",
    2: "ReLU",
    3: "Clip",
    4: "Sigmoid",
    5: "Mish",
    6: "HardSwish",
}

PARAM_TOKEN = re.compile(r"^[+-]?\d+=")


def known_layers(sources: Path) -> list[str]:
    text = sources.read_text(encoding="utf-8")
    return re.findall(r'\.\{ \.name = "([^"]+)"', text)


def parse_param(text: str) -> tuple[list[str], set[str]]:
    tokens = text.split()
    pos = 0

    def next_token() -> str:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("truncated param file")
        token = tokens[pos]
        pos += 1
        return token

    def next_int(label: str) -> int:
        try:
            return int(next_token())
        except ValueError:
            raise ValueError(f"invalid {label}") from None

    magic = next_int("magic")
    if magic != 7767517:
        raise ValueError("invalid magic; this is not an ncnn param file")

    layer_count = next_int("layer count")
    blob_count = next_int("blob count")
    if layer_count <= 0 or blob_count <= 0:
        raise ValueError("layer_count and blob_count must be positive")

    layers: list[str] = []
    activations: set[str] = set()

    for _ in range(layer_count):
        layer_type = next_token()
        if PARAM_TOKEN.match(layer_type):
            raise ValueError("expected a layer header")
        _ = next_token()  # layer name
        bottom_count = next_int("bottom_count")
        top_count = next_int("top_count")
        if bottom_count < 0 or top_count < 0:
            raise ValueError("negative blob count")

        for _ in range(bottom_count + top_count):
            _ = next_token()

        activation_type = 0
        while pos < len(tokens) and PARAM_TOKEN.match(tokens[pos]):
            token = tokens[pos]
            pos += 1
            if layer_type not in FUSED_OWNERS:
                continue
            key, sep, value = token.partition("=")
            if sep and key == "9":
                try:
                    activation_type = int(value)
                except ValueError:
                    pass

        if layer_type in FUSED_OWNERS:
            activation = ACTIVATIONS.get(activation_type)
            if activation:
                activations.add(activation)

        layers.append(layer_type)

    if pos != len(tokens):
        raise ValueError("trailing data after the declared layers")

    return layers, activations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print a runnable ncnn-zig .layers suggestion from a .param file."
    )
    parser.add_argument("param", type=Path, help="path to model.param")
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "src" / "sources.zig",
        help="path to src/sources.zig (default: the ncnn-zig checkout)",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="omit default helper layers (you must add the ones your app needs)",
    )
    args = parser.parse_args()

    try:
        known = known_layers(args.sources)
        known_set = set(known)
        layers, activations = parse_param(args.param.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        print(f"layers.py: {err}", file=sys.stderr)
        return 1

    custom = [name for name in dict.fromkeys(layers) if name not in known_set]
    model = [name for name in dict.fromkeys(layers) if name in known_set]

    selected = set(model) | activations
    if not args.minimal:
        selected.update(DEFAULT_HELPERS)

    missing = sorted(selected - known_set)
    if missing:
        print(f"layers.py: unknown helper layer(s): {', '.join(missing)}", file=sys.stderr)
        return 1

    ordered = [name for name in known if name in selected]
    print('.layers = "' + ",".join(ordered) + '",')

    if activations:
        print(
            "# fused activations added: " + ", ".join(sorted(activations)),
            file=sys.stderr,
        )
    if custom:
        print(
            "# custom layers need runtime registration: "
            + ", ".join(custom),
            file=sys.stderr,
        )
    if args.minimal:
        print(
            "# --minimal: this may crash with fp16/bf16 weights or Mat/pixel helpers",
            file=sys.stderr,
        )
    else:
        print(
            "# helper layers included: " + ", ".join(DEFAULT_HELPERS),
            file=sys.stderr,
        )
        print(
            "# Cast is included because fp16/bf16 weights load through it",
            file=sys.stderr,
        )
    print(
        "# ncnn-zig expands dependencies such as Convolution -> Gemm automatically",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
