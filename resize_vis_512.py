#!/usr/bin/env python3
"""
将单张图片缩放到 512×512，再缩回原尺寸，各保存一份。

用法示例:
  python Fusion5/resize_vis_512.py "path/to/image.png"
  python Fusion5/resize_vis_512.py -i image.png -o512 out_512.png -o_orig out_orig.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def _load_rgb(path: Path) -> Image.Image:
    im = Image.open(path)
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    elif im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        im = bg
    return im


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resize to 512×512 and back to original size; save both."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="输入图片路径（可省略，改用 -i）",
    )
    parser.add_argument(
        "-i",
        "--input-path",
        default=None,
        help="输入图片路径（与位置参数二选一）",
    )
    parser.add_argument(
        "-o512",
        "--output-512",
        default=None,
        help="512 版本输出路径；默认 原名_512.png",
    )
    parser.add_argument(
        "-o_orig",
        "--output-orig",
        default=None,
        help="缩回原尺寸输出路径；默认 原名_restored.png",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="中间缩放宽，默认 512",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="中间缩放高，默认 512",
    )
    parser.add_argument(
        "--method",
        choices=("lanczos", "bicubic", "nearest"),
        default="lanczos",
        help="插值方式",
    )
    args = parser.parse_args()

    inp = args.input_path or args.input
    if inp is None:
        parser.error("请提供输入图片路径，例如:\n  python resize_vis_512.py path/to/image.png")

    src = Path(inp).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"找不到文件: {src}")

    resample = {
        "lanczos": Image.Resampling.LANCZOS,
        "bicubic": Image.Resampling.BICUBIC,
        "nearest": Image.Resampling.NEAREST,
    }[args.method]

    im = _load_rgb(src)
    orig_size = im.size  # (W, H)

    mid_hw = (args.width, args.height)
    im_512 = im.resize(mid_hw, resample)
    im_restored = im_512.resize(orig_size, resample)

    if args.output_512:
        dst_512 = Path(args.output_512).expanduser().resolve()
    else:
        dst_512 = src.parent / f"{src.stem}_512{src.suffix}"

    if args.output_orig:
        dst_orig = Path(args.output_orig).expanduser().resolve()
    else:
        dst_orig = src.parent / f"{src.stem}_restored{src.suffix}"

    dst_512.parent.mkdir(parents=True, exist_ok=True)
    dst_orig.parent.mkdir(parents=True, exist_ok=True)

    im_512.save(dst_512, quality=95)
    im_restored.save(dst_orig, quality=95)

    print(f"输入:     {src}  ({orig_size[0]}×{orig_size[1]})")
    print(f"512 版:   {dst_512}  ({mid_hw[0]}×{mid_hw[1]})")
    print(f"还原版:   {dst_orig}  ({orig_size[0]}×{orig_size[1]})")


if __name__ == "__main__":
    main()
