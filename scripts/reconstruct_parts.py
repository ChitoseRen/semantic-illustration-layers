# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 ChitoseRen
"""Reconstruct full-canvas candidates from protected source pixels and registered patches.

Usage: python reconstruct_parts.py reconstruction.json
Dependencies: Pillow and numpy. This tool does not paint or infer hidden structure.
Inputs are interpreted as sRGB; embedded ICC profiles must be handled beforehand.
Color fields use normalized smoothing with three separable box filters approximating
a Gaussian. Sigma is in original-canvas pixels; support is the smoothed sample weight.
An exit code of zero means a candidate needs visual review, never artistic acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
import uuid

import numpy as np
from PIL import Image

from asset_pipeline import registration_matrix, warp_rgba


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def srgb_to_linear(rgb):
    rgb = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb):
    rgb = np.clip(rgb, 0.0, 1.0)
    encoded = np.where(rgb <= 0.0031308, rgb * 12.92,
                       1.055 * rgb ** (1.0 / 2.4) - 0.055)
    return np.asarray(np.clip(encoded * 255.0 + 0.5, 0, 255), dtype=np.uint8)


def _identifier(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[\w-]+", value):
        raise ValueError(label + " must contain letters, numbers, underscores or hyphens")
    if value.upper() in {"CON", "PRN", "AUX", "NUL", *("COM" + str(i) for i in range(1, 10)),
                         *("LPT" + str(i) for i in range(1, 10))}:
        raise ValueError(label + " is a reserved Windows filename")
    return value


def _finite_number(value, label, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(label + " must be a finite number")
    if (positive and value <= 0) or (not positive and value < 0):
        raise ValueError(label + (" must be positive" if positive else " must be nonnegative"))
    return float(value)


class Inputs:
    """Read and hash the exact bytes consumed, with project-contained paths."""

    def __init__(self, filename):
        self.filename = Path(filename).resolve()
        self.root = self.filename.parent
        payload = self.filename.read_bytes()
        self.hashes = {self.filename.name: sha256(payload)}
        self.paths = {self.filename}
        self.config = json.loads(payload.decode("utf-8-sig"))
        if not isinstance(self.config, dict):
            raise ValueError("Configuration must be a JSON object")

    def resolve(self, value):
        if not isinstance(value, str) or not value or Path(value).is_absolute():
            raise ValueError("All paths must be nonempty paths relative to the configuration")
        path = (self.root / value).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Path escapes the configuration directory: " + value)
        return path

    def image(self, value, label, size=None, mode=None, binary=False):
        path = self.resolve(value)
        payload = path.read_bytes()
        self.paths.add(path)
        self.hashes[path.relative_to(self.root).as_posix()] = sha256(payload)
        with Image.open(io.BytesIO(payload)) as image:
            if image.info.get("icc_profile"):
                raise ValueError(label + " has an embedded ICC profile; convert to sRGB and remove it first")
            if mode and image.mode != mode:
                raise ValueError(label + " must be " + mode + ", got " + image.mode)
            if not mode and image.mode not in ("RGB", "RGBA"):
                raise ValueError(label + " must be RGB or RGBA in sRGB")
            if size and image.size != size:
                raise ValueError(label + " must match the original canvas size")
            result = image.copy()
        if binary and not np.isin(np.asarray(result), (0, 255)).all():
            raise ValueError(label + " must contain only 0 and 255")
        return result

    def mask(self, value, label, size, binary=True):
        return np.asarray(self.image(value, label, size, "L", binary)).copy()


def _box_blur_axis(array, radius, axis):
    if radius == 0:
        return array.copy()
    moved = np.moveaxis(array, axis, 0)
    prefix = np.concatenate((np.zeros_like(moved[:1], dtype=np.float64),
                             np.cumsum(moved, axis=0, dtype=np.float64)), axis=0)
    positions = np.arange(len(moved), dtype=np.int64)
    low = np.maximum(0, positions - radius)
    high = np.minimum(len(moved), positions + radius + 1)
    result = (prefix[high] - prefix[low]) / (2 * radius + 1)
    return np.moveaxis(result, 0, axis).astype(np.float32)


def gaussian_box_widths(sigma):
    """Three odd box widths whose total variance approximates sigma squared."""
    ideal = math.sqrt(4 * sigma * sigma + 1)
    lower = max(1, int(math.floor(ideal)))
    if lower % 2 == 0:
        lower -= 1
    upper = lower + 2
    count_lower = round((12 * sigma * sigma - 3 * lower * lower - 12 * lower - 9)
                        / (-4 * lower - 4))
    count_lower = min(3, max(0, count_lower))
    return [lower] * count_lower + [upper] * (3 - count_lower)


def gaussian_blur(array, sigma):
    result = np.asarray(array, dtype=np.float32)
    for width in gaussian_box_widths(sigma):
        radius = width // 2
        result = _box_blur_axis(_box_blur_axis(result, radius, 0), radius, 1)
    return result


def _load_parts(inputs, source):
    size = source.size
    source_array = np.asarray(source.convert("RGBA"))
    parts = inputs.config.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("parts must be a nonempty list")
    loaded, folded_ids = {}, set()
    for index, spec in enumerate(parts):
        pid = _identifier(spec["id"], "Part id")
        if pid.casefold() in folded_ids:
            raise ValueError("Duplicate part id ignoring case: " + pid)
        folded_ids.add(pid.casefold())
        order = spec.get("order", index)
        if isinstance(order, bool) or not isinstance(order, (int, float)) or not math.isfinite(order):
            raise ValueError(pid + ": order must be finite")
        layer = np.asarray(inputs.image(spec["source_layer"], pid + ".source_layer", size, "RGBA")).copy()
        masks = {key: inputs.mask(spec[key], pid + "." + key, size,
                                  binary=key not in ("target_alpha", "generated_weight"))
                 for key in ("keep", "unknown", "edit", "target_alpha", "generated_weight", "blend_allowed")}
        masks["color_protect"] = (inputs.mask(spec["color_protect"], pid + ".color_protect", size)
                                    if spec.get("color_protect") else np.zeros(size[::-1], dtype=np.uint8))
        candidate_image = inputs.image(spec["candidate"], pid + ".candidate", mode="RGBA")
        matrix, errors, determinant = registration_matrix(spec.get("registration", {}), candidate_image.size, size)
        candidate = np.asarray(warp_rgba(candidate_image, matrix, size)).copy()
        keep, unknown, edit = (masks[key] > 0 for key in ("keep", "unknown", "edit"))
        target, weight = masks["target_alpha"], masks["generated_weight"]
        active = target > 0
        generated = weight > 0
        blend = (weight > 0) & (weight < 255)
        if np.any(keep & (unknown | edit)):
            raise ValueError(pid + ": keep overlaps unknown or edit")
        if np.any(keep & ((layer[..., 3] != 255) | (source_array[..., 3] != 255) | (target != 255))):
            raise ValueError(pid + ": keep requires opaque source, source_layer and target_alpha")
        if np.any(keep[..., None] & (layer[..., :3] != source_array[..., :3])):
            raise ValueError(pid + ": source_layer RGB differs from source in keep")
        if np.any(generated & (~edit | keep)):
            raise ValueError(pid + ": nonzero generated_weight must be inside edit and outside keep")
        if np.any(generated & ~active):
            raise ValueError(pid + ": generated_weight must be zero outside target_alpha")
        if np.any(blend & ((masks["blend_allowed"] != 255) | (target != 255)
                          | (layer[..., 3] != 255) | (candidate[..., 3] != 255))):
            raise ValueError(pid + ": intermediate weights require blend_allowed and three opaque alphas")
        if np.any(active & (weight < 255) & (layer[..., 3] < target)):
            raise ValueError(pid + ": selected source pixels cannot support target_alpha")
        if np.any(active & generated & (candidate[..., 3] < target)):
            raise ValueError(pid + ": selected candidate pixels cannot support target_alpha")
        loaded[pid] = {"id": pid, "order": order, "source": layer,
                       "candidate": candidate, "corrected": candidate[..., :3].copy(),
                       "masks": masks, "transform": matrix.tolist(),
                       "landmark_errors_px": errors, "determinant": determinant,
                       "candidate_size": list(candidate_image.size)}
    return loaded


def _load_groups(inputs, parts, size):
    groups = inputs.config.get("appearance_groups", [])
    if not isinstance(groups, list):
        raise ValueError("appearance_groups must be a list")
    loaded, ids, target_regions = [], set(), {}
    for spec in groups:
        gid = _identifier(spec["id"], "Appearance group id")
        if gid.casefold() in ids:
            raise ValueError("Duplicate appearance group id ignoring case: " + gid)
        ids.add(gid.casefold())
        region = inputs.mask(spec["region"], gid + ".region", size) > 0
        if not np.any(region):
            raise ValueError(gid + ": region is empty")
        targets = spec["targets"]
        if not isinstance(targets, list) or not targets or len(set(targets)) != len(targets):
            raise ValueError(gid + ": targets must be a nonempty list of unique part ids")
        for pid in targets:
            if pid not in parts:
                raise ValueError(gid + ": unknown target part " + str(pid))
            previous = target_regions.setdefault(pid, np.zeros(size[::-1], dtype=bool))
            if np.any(previous & region):
                raise ValueError(gid + ": appearance group regions overlap for target " + pid)
            previous |= region
        sigma = _finite_number(spec["sigma_px"], gid + ".sigma_px", positive=True)
        min_support = _finite_number(spec["min_support"], gid + ".min_support", positive=True)
        max_correction = _finite_number(spec["max_correction"], gid + ".max_correction")
        samples = spec["samples"]
        if not isinstance(samples, list):
            raise ValueError(gid + ": samples must be a list")
        sample_masks = {}
        for sample in samples:
            pid = sample["part"]
            if pid not in parts:
                raise ValueError(gid + ": unknown sample part " + str(pid))
            mask = inputs.mask(sample["mask"], gid + ".sample." + pid, size) > 0
            part = parts[pid]
            allowed = (region & (part["masks"]["keep"] == 255)
                       & (part["candidate"][..., 3] == 255)
                       & (part["masks"]["unknown"] == 0)
                       & (part["masks"]["color_protect"] == 0))
            if np.any(mask & ~allowed):
                raise ValueError(gid + ": samples for " + pid
                                 + " must be inside region and keep, opaque in candidate, outside unknown/color_protect")
            sample_masks.setdefault(pid, np.zeros(size[::-1], dtype=bool))[:] |= mask
        loaded.append({"id": gid, "region": region, "targets": targets, "samples": sample_masks,
                       "sigma_px": sigma, "min_support": min_support, "max_correction": max_correction})
    return loaded


def _apply_groups(groups, parts, size):
    records, support_images, diagnostics = [], {}, []
    for group in groups:
        gid = group["id"]
        target_records = []
        for pid in group["targets"]:
            part = parts[pid]
            # Parts share an original-image surface definition, never another
            # candidate's residual: opposite candidate biases must not cancel.
            sample_mask = group["samples"].get(pid, np.zeros(size[::-1], dtype=bool))
            support = gaussian_blur(sample_mask.astype(np.float32), group["sigma_px"])
            residual = np.zeros((*size[::-1], 3), dtype=np.float32)
            residual[sample_mask] = (srgb_to_linear(part["source"][..., :3][sample_mask])
                                     - srgb_to_linear(part["candidate"][..., :3][sample_mask]))
            field = np.stack([gaussian_blur(residual[..., channel], group["sigma_px"])
                              for channel in range(3)], axis=-1)
            np.divide(field, support[..., None], out=field, where=support[..., None] > 0)
            supported = (support >= group["min_support"]) & (support > 0)
            bounded = np.max(np.abs(field), axis=-1) <= group["max_correction"]
            selected = (group["region"] & (part["masks"]["generated_weight"] > 0)
                        & (part["masks"]["target_alpha"] > 0)
                        & (part["masks"]["color_protect"] == 0))
            unsupported = selected & ~supported
            excessive = selected & supported & ~bounded
            accepted = selected & supported & bounded
            if np.any(unsupported):
                diagnostics.append({"group": gid, "part": pid, "reason": "insufficient_sample_support",
                                    "pixels": int(unsupported.sum())})
            if np.any(excessive):
                diagnostics.append({"group": gid, "part": pid, "reason": "max_correction_exceeded",
                                    "pixels": int(excessive.sum())})
            corrected = srgb_to_linear(part["candidate"][..., :3][accepted]) + field[accepted]
            clipped = np.any((corrected < 0) | (corrected > 1), axis=-1)
            part["corrected"][accepted] = linear_to_srgb(corrected)
            target_records.append({"part": pid, "applicable_pixels": int(selected.sum()),
                                   "corrected_pixels": int(accepted.sum()),
                                   "unsupported_pixels": int(unsupported.sum()),
                                   "excessive_correction_pixels": int(excessive.sum()),
                                   "gamut_clipped_pixels": int(clipped.sum()),
                                   "sample_pixels": int(sample_mask.sum()),
                                   "support_file": "support/" + gid + "/" + pid + ".png",
                                   "support_min": float(support.min()), "support_max": float(support.max()),
                                   "minimum_target_support": float(support[selected].min()) if np.any(selected) else None})
            # 16-bit support: 65535 = one fully weighted sample per pixel.
            support_images[(gid, pid)] = np.uint16(np.clip(support, 0, 1) * 65535 + 0.5)
        records.append({"id": gid, "status": "incomplete" if any(
            d["group"] == gid for d in diagnostics) else "candidate_needs_visual_review",
                        "sample_pixels_per_part": {pid: int(mask.sum()) for pid, mask in group["samples"].items()},
                        "sigma_px": group["sigma_px"], "box_widths_px": gaussian_box_widths(group["sigma_px"]),
                        "min_support": group["min_support"], "max_correction": group["max_correction"],
                        "targets": target_records})
    return records, support_images, diagnostics


def _assemble(part, corrected):
    source, target, weight = part["source"], part["masks"]["target_alpha"], part["masks"]["generated_weight"]
    candidate_rgb = part["corrected"] if corrected else part["candidate"][..., :3]
    result = np.zeros_like(source)
    result[..., 3] = target
    source_pixels = (target > 0) & (weight == 0)
    candidate_pixels = (target > 0) & (weight == 255)
    blended = (target > 0) & (weight > 0) & (weight < 255)
    result[..., :3][source_pixels] = source[..., :3][source_pixels]
    result[..., :3][candidate_pixels] = candidate_rgb[candidate_pixels]
    if np.any(blended):
        amount = weight[blended, None].astype(np.float32) / 255.0
        mixed = (srgb_to_linear(source[..., :3][blended]) * (1 - amount)
                 + srgb_to_linear(candidate_rgb[blended]) * amount)
        result[..., :3][blended] = linear_to_srgb(mixed)
    return Image.fromarray(result, "RGBA")


def _check_output_paths(inputs, out, targets):
    if out == inputs.root:
        raise ValueError("output must be a dedicated directory inside the configuration directory")
    if any(path.is_relative_to(out) for path in inputs.paths):
        raise ValueError("Input/output overlap: the output directory contains an input")
    for target in targets:
        resolved = target.resolve()
        if not resolved.is_relative_to(out) or not resolved.is_relative_to(inputs.root):
            raise ValueError("An output path escapes its declared directory")
        if resolved in inputs.paths or (target.exists() and any(target.samefile(path) for path in inputs.paths)):
            raise ValueError("An output would overwrite an input")


def reconstruct(filename):
    inputs = Inputs(filename)
    # Import lazily so --help remains available independently of the preflight tool.
    from mask_preflight import inspect_config
    preflight = inspect_config(inputs.filename)
    if preflight.get("errors"):
        raise ValueError("Mask preflight has errors: " + json.dumps(preflight["errors"], ensure_ascii=False))
    fingerprint = preflight.get("fingerprint")
    review = inputs.config.get("preflight_review", {})
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("Mask preflight did not return a fingerprint")
    if not isinstance(review, dict) or (review.get("status") != "reviewed"
            or review.get("fingerprint") != fingerprint
            or not isinstance(review.get("evidence"), str) or not review["evidence"].strip()):
        raise ValueError("A current preflight_review with reviewed status, matching fingerprint and actual evidence is required")
    source = inputs.image(inputs.config["source"], "source")
    size = source.size
    parts = _load_parts(inputs, source)
    groups = _load_groups(inputs, parts, size)
    output_root = inputs.resolve(inputs.config.get("output", "reconstruction"))
    out = output_root / "runs" / uuid.uuid4().hex
    targets = ([out / "parts" / (pid + ".png") for pid in parts]
               + [out / "baseline_parts" / (pid + ".png") for pid in parts]
               + [out / "support" / group["id"] / (pid + ".png") for group in groups for pid in group["targets"]]
               + [out / name for name in ("before.png", "after.png", "reconstruction-report.json")])
    _check_output_paths(inputs, output_root, targets)
    group_records, support_images, diagnostics = _apply_groups(groups, parts, size)
    before = Image.new("RGBA", size)
    after = Image.new("RGBA", size)
    records = []
    out.mkdir(parents=True, exist_ok=True)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    (out / "baseline_parts").mkdir(parents=True, exist_ok=True)
    ordered = sorted(parts.values(), key=lambda part: part["order"])
    for part in ordered:
        original = _assemble(part, corrected=False)
        result = _assemble(part, corrected=True)
        before.alpha_composite(original)
        after.alpha_composite(result)
        result_path = out / "parts" / (part["id"] + ".png")
        baseline_path = out / "baseline_parts" / (part["id"] + ".png")
        original.save(baseline_path)
        result.save(result_path)
        selected = part["masks"]["keep"] == 255
        keep_different = int(np.any(np.asarray(result)[..., :3] != part["source"][..., :3], axis=-1)[selected].sum())
        records.append({"id": part["id"], "order": part["order"],
                        "file": result_path.relative_to(inputs.root).as_posix(),
                        "baseline_file": baseline_path.relative_to(inputs.root).as_posix(),
                        "baseline_sha256": sha256(baseline_path.read_bytes()),
                        "color_selection": "unselected_requires_comparison",
                        "status": "incomplete" if any(d["part"] == part["id"] for d in diagnostics)
                        else "candidate_needs_visual_review",
                        "output_sha256": sha256(result_path.read_bytes()),
                        "keep_pixels": int(selected.sum()), "keep_rgb_changed_pixels": keep_different,
                        "alpha_unchanged_by_color_correction": bool(np.array_equal(
                            np.asarray(original)[..., 3], np.asarray(result)[..., 3])),
                        "transform_candidate_to_canvas": part["transform"],
                        "landmark_errors_px": part["landmark_errors_px"], "determinant": part["determinant"],
                        "candidate_size": part["candidate_size"],
                        "coordinate_space": "source_canvas",
                        "production_registration": {"frame": [0, 0, *size]}})
    before.save(out / "before.png")
    after.save(out / "after.png")
    if support_images:
        (out / "support").mkdir(parents=True, exist_ok=True)
        for (gid, pid), support in support_images.items():
            (out / "support" / gid).mkdir(parents=True, exist_ok=True)
            Image.fromarray(support).save(out / "support" / gid / (pid + ".png"))
    hash_manifest = dict(sorted(inputs.hashes.items()))
    for group in group_records:
        for target in group["targets"]:
            target["support_file"] = (out / target["support_file"]).relative_to(inputs.root).as_posix()
    report = {"status": "incomplete" if diagnostics else "candidate_needs_visual_review",
              "color_selection": "unselected_requires_comparison",
              "run_directory": out.relative_to(inputs.root).as_posix(),
              "report": (out / "reconstruction-report.json").relative_to(inputs.root).as_posix(),
              "canvas": list(size), "preflight_fingerprint": fingerprint,
              "input_sha256": hash_manifest,
              "fingerprint": sha256(json.dumps(hash_manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")),
              "color_space": "sRGB interpreted explicitly; residuals and internal blends in linear sRGB",
              "smoothing": "normalized three-box Gaussian approximation, zero boundary, independent residual/support fields per (appearance group, target part)",
              "support_image_scale": "16-bit grayscale: 65535 equals sample weight 1; values above 1 saturate in the image only",
              "parts": records, "appearance_groups": group_records, "diagnostics": diagnostics,
              "before": (out / "before.png").relative_to(inputs.root).as_posix(),
              "after": (out / "after.png").relative_to(inputs.root).as_posix(),
              "composites": "transparent RGBA; appearance_check requires separate composites over the same opaque background",
              "note": "No visual, ownership, hidden-structure or motion acceptance is inferred. Color correction is not selected automatically: compare the uncorrected baseline and corrected candidate with independent local evidence before choosing. Incomplete candidates must not enter production."}
    (out / "reconstruction-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Project-local reconstruction JSON with a current preflight review")
    args = parser.parse_args()
    try:
        report = reconstruct(args.config)
        print(json.dumps({"status": report["status"], "part_count": len(report["parts"]),
                          "diagnostic_count": len(report["diagnostics"]),
                          "report": str(args.config.resolve().parent / report["report"])}, ensure_ascii=False))
        return 2 if report["status"] == "incomplete" else 0
    except (ValueError, OSError, KeyError, TypeError, IndexError, OverflowError, np.linalg.LinAlgError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
