# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 ChitoseRen
"""Global chromatic-matte alpha extraction and narrow-edge unmixing only.

Dependencies: Pillow, numpy. Does not generate artwork or fill enclosed holes.
Always inspect semantic colour conflicts and black/white QA before accepting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def parse_color(value):
    named = {"green": (0, 255, 0), "blue": (0, 0, 255), "red": (255, 0, 0),
             "cyan": (0, 255, 255), "magenta": (255, 0, 255), "yellow": (255, 255, 0)}
    value = value.strip().lower()
    if value in named:
        return named[value]
    if value.startswith("#") and len(value) == 7:
        return tuple(int(value[i:i+2], 16) for i in (1, 3, 5))
    if value.startswith("rgb(") and value.endswith(")"):
        value = value[4:-1]
    try:
        rgb = tuple(int(x.strip()) for x in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use green, blue, R,G,B or #RRGGBB") from error
    if len(rgb) != 3 or min(rgb) < 0 or max(rgb) > 255:
        raise argparse.ArgumentTypeError("RGB needs three integer channels from 0 to 255")
    return rgb


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_protect(path, size):
    if path is None:
        return np.zeros((size[1], size[0]), dtype=bool)
    image = Image.open(path)
    if image.size != size:
        raise ValueError("Protect mask must match input canvas exactly")
    if "A" in image.getbands():
        channel = image.getchannel("A")
    elif image.mode in ("1", "L"):
        channel = image.convert("L")
    else:
        rgb = np.asarray(image.convert("RGB"))
        if not np.array_equal(rgb[..., 0], rgb[..., 1]) or not np.array_equal(rgb[..., 0], rgb[..., 2]):
            raise ValueError("Protect must be a grayscale mask or explicit RGBA alpha")
        channel = image.convert("L")
    return np.asarray(channel) >= 128


def extract(original, matte, protect, key_excess=60, key_level=130, edge_excess=6, edge_radius=5):
    source = np.asarray(original.convert("RGBA"))
    rgb = source[..., :3].copy()
    f, color = rgb.astype(np.float32), np.asarray(matte, dtype=np.float32)
    chroma = float(color.max()-color.min())
    if chroma < 32:
        raise ValueError("This chromatic-matte method needs a saturated key color; neutral/near-gray mattes require an inspected mask/trimap, not brightness keying")
    high = color >= color.min()+chroma*.5
    low = ~high
    matte_excess = float(color[high].min()-color[low].max())
    if matte_excess < 16:
        raise ValueError("Matte RGB channel separation is too small for this chroma rule")
    # Pure green reduces exactly to g-max(r,b), pure blue to b-max(r,g).
    # For two-high-channel keys such as magenta, use min(r,b)-g.
    excess = (f[..., high].min(axis=-1)-f[..., low].max(axis=-1))*255/matte_excess
    level = f[..., high].min(axis=-1)
    candidates = (level > key_level*float(color[high].min())/255) & (excess > key_excess)
    keyed = candidates & ~protect
    keyed_image = Image.fromarray(keyed.astype(np.uint8)*255, "L")
    near = np.asarray(keyed_image.filter(ImageFilter.MaxFilter(edge_radius*2+1))) > 0 if edge_radius else keyed
    edge = near & ~keyed & ~protect & (excess > edge_excess) & (source[..., 3] > 0)
    key_alpha = np.ones(keyed.shape, dtype=np.float32)
    key_alpha[keyed] = 0
    estimated_alpha = np.clip(1-excess/255, 0, 1)
    key_alpha[edge] = estimated_alpha[edge]
    safe = np.maximum(estimated_alpha, .001)
    # Standard inverse C = alpha*F + (1-alpha)*matte. Assumes no target-key
    # chroma at this narrow edge; semantic conflicts must be protected/rejected.
    clean = (f-(1-estimated_alpha[..., None])*color)/safe[..., None]
    rgb[edge] = np.clip(np.rint(clean[edge]), 0, 255).astype(np.uint8)
    alpha = np.rint(key_alpha*source[..., 3]).astype(np.uint8)
    result = np.dstack((rgb, alpha))
    changed = np.any(rgb != source[..., :3], axis=-1)
    if np.any(changed & ~edge):
        raise AssertionError("RGB changed outside the declared narrow edge")
    if not np.array_equal(result[protect], source[protect]):
        raise AssertionError("Protected RGBA changed")
    report = {
        "method": "global chromatic key including enclosed holes; narrow retained-edge inverse matte mixture",
        "matte_rgb": list(matte), "high_channels": np.flatnonzero(high).tolist(),
        "low_channels": np.flatnonzero(low).tolist(), "matte_channel_excess": matte_excess,
        "key_excess_normalized_255": key_excess, "key_level_normalized_255": key_level,
        "edge_excess_normalized_255": edge_excess, "edge_radius_input_pixels": edge_radius,
        "key_candidates_before_protection": int(candidates.sum()),
        "protected_candidate_pixels": int((candidates & protect).sum()),
        "removed_matte_pixels": int(keyed.sum()), "edge_unmixed_pixels": int(edge.sum()),
        "edge_rgb_changed_pixels": int(changed.sum()), "non_edge_rgb_changed_pixels": 0,
        "protected_rgba_changed_pixels": 0,
        "opaque_pixels": int((alpha == 255).sum()),
        "partial_alpha_pixels": int(((alpha > 0) & (alpha < 255)).sum()),
        "transparent_pixels": int((alpha == 0).sum()),
        "topology": "global keying; no flood fill, hole fill, closing, component deletion or silhouette painting",
        "semantic_colour_conflict_check": "required: inspect whether any intended foreground uses the matte hue; preserve it with an observed protect mask or choose a different matte. The algorithm does not prove absence of colour conflicts.",
        "visual_status": "requires_semantic_colour_check_black_white_review_and_final_composite",
        "artwork_generation_calls": 0,
    }
    return Image.fromarray(result, "RGBA"), keyed_image, Image.fromarray(edge.astype(np.uint8)*255, "L"), report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matte-color", type=parse_color, required=True)
    parser.add_argument("--protect", type=Path, help="Optional observed foreground protection mask; white>=128 preserves original RGBA")
    parser.add_argument("--key-excess", type=float, default=60)
    parser.add_argument("--key-level", type=float, default=130)
    parser.add_argument("--edge-excess", type=float, default=6)
    parser.add_argument("--edge-radius", type=int, default=5, help="Narrow despill band in input pixels; 0 disables RGB unmixing")
    args = parser.parse_args()
    if not (0 <= args.edge_radius <= 20):
        parser.error("--edge-radius must be from 0 to 20 input pixels")
    if not (0 <= args.edge_excess < args.key_excess < 255 and 0 <= args.key_level < 255):
        parser.error("Require 0<=edge-excess<key-excess<255 and 0<=key-level<255")
    if args.input.resolve() == args.output.resolve() or args.protect and args.protect.resolve() == args.output.resolve():
        parser.error("Output must not overwrite input or protection mask")
    if args.output.suffix.lower() != ".png":
        parser.error("RGBA output must use .png")
    prefix = args.output.with_suffix("")
    companions = [Path(str(prefix)+suffix) for suffix in ("-alpha.png", "-keyed.png", "-edge-band.png", "-on-black.png", "-on-white.png", "-black-white.png", "-report.json")]
    inputs = {args.input.resolve()} | ({args.protect.resolve()} if args.protect else set())
    if any(path.resolve() in inputs for path in companions):
        parser.error("A QA companion would overwrite the input/protect mask; choose a different output name")
    original = Image.open(args.input)
    protect = read_protect(args.protect, original.size)
    result, keyed, edge, report = extract(original, args.matte_color, protect,
        args.key_excess, args.key_level, args.edge_excess, args.edge_radius)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    result.getchannel("A").save(str(prefix)+"-alpha.png")
    keyed.save(str(prefix)+"-keyed.png")
    edge.save(str(prefix)+"-edge-band.png")
    previews = []
    for name, color in (("black", (0, 0, 0, 255)), ("white", (255, 255, 255, 255))):
        image = Image.new("RGBA", result.size, color)
        image.alpha_composite(result)
        image = image.convert("RGB")
        image.save(str(prefix)+"-on-"+name+".png")
        image.thumbnail((560, 1000), Image.Resampling.LANCZOS)
        previews.append(image)
    contact = Image.new("RGB", (sum(im.width for im in previews), max(im.height for im in previews)), (90, 90, 90))
    x = 0
    for image in previews:
        contact.paste(image, (x, 0))
        x += image.width
    contact.save(str(prefix)+"-black-white.png")
    report.update(input=str(args.input), output=str(args.output), size=list(result.size),
                  input_sha256=sha256(args.input), output_sha256=sha256(args.output),
                  protect=str(args.protect) if args.protect else None,
                  protect_sha256=sha256(args.protect) if args.protect else None,
                  alpha_bbox=result.getchannel("A").getbbox())
    Path(str(prefix)+"-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
