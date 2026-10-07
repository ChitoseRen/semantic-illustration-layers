# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 ChitoseRen
"""Diagnostic checks for aligned, opaque illustration composites.

Usage: python appearance_check.py appearance-review.json
Requires Pillow and numpy only. Measurements do not imply a visual pass.
All paths are relative to the configuration directory; output must be new.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


MAX_PIXELS = 20_000_000
MAX_DIMENSION = 16_384
MAX_REGIONS = 64
MAX_PROFILES = 64
MAX_PROFILE_SAMPLES = 100_000
MAX_OUTPUT_PIXELS = 300_000_000
RGB_UNIT = "encoded sRGB code values, 0..255; engineering difference, not perceptual Delta E"
LUMA_UNIT = "linear sRGB relative luminance, 0..1"
ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = value
    return result


def finite_number(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def local_path(root, value, field, *, output=False):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}: a nonempty relative path is required")
    relative = Path(value)
    if relative.is_absolute() or relative.drive:
        raise ValueError(f"{field}: absolute paths are not allowed")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"{field}: path must remain inside the configuration directory")
    if output:
        if path.exists():
            raise ValueError(f"{field}: output already exists; choose a new directory")
    elif not path.is_file():
        raise ValueError(f"{field}: input file does not exist")
    return path


def read_image(path, field, *, size=None, mask=False, binary=False):
    with Image.open(path) as image:
        width, height = image.size
        if (width < 1 or height < 1 or width * height > MAX_PIXELS
                or max(width, height) > MAX_DIMENSION):
            raise ValueError(f"{field}: canvas exceeds supported bounds")
        if image.info.get("icc_profile"):
            raise ValueError(f"{field}: embedded ICC profile; explicitly convert inputs to untagged sRGB first")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"{field}: multi-frame images are not supported")
        if size is not None and image.size != size:
            raise ValueError(f"{field}: canvas size {image.size} differs from source {size}")
        if mask:
            if image.mode != "L":
                raise ValueError(f"{field}: mask must use L mode")
            array = np.asarray(image).copy()
            if binary and np.any((array != 0) & (array != 255)):
                raise ValueError(f"{field}: binary mask must contain only 0 and 255")
            return array
        if image.mode not in ("RGB", "RGBA"):
            raise ValueError(f"{field}: composite must use RGB or opaque RGBA mode")
        if "transparency" in image.info:
            raise ValueError(f"{field}: composite contains transparency metadata; flatten explicitly first")
        if image.mode == "RGBA" and image.getchannel("A").getextrema() != (255, 255):
            raise ValueError(f"{field}: RGBA composite must be fully opaque")
        return image.convert("RGB").copy()


def validate_id(value, field, seen):
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ValueError(f"{field}: id must use 1..80 ASCII letters, digits, underscores or hyphens")
    if value.lower() in seen:
        raise ValueError(f"{field}: duplicate id (case-insensitive)")
    seen.add(value.lower())


def profile_coordinates(points):
    """Uniform arc-length samples, at most one source pixel apart."""
    points = np.asarray(points, dtype=np.float64)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(lengths <= 0):
        raise ValueError("profiles: adjacent points must differ")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    count = int(math.ceil(float(cumulative[-1]))) + 1
    if count > MAX_PROFILE_SAMPLES:
        raise ValueError("profiles: one profile exceeds the sample limit")
    distance = np.linspace(0.0, cumulative[-1], count)
    segment = np.minimum(np.searchsorted(cumulative, distance, side="right") - 1, len(lengths) - 1)
    fraction = (distance - cumulative[segment]) / lengths[segment]
    coordinates = points[segment] + fraction[:, None] * (points[segment + 1] - points[segment])
    return distance, coordinates


def load_config(config_path):
    config_path = Path(config_path).resolve(strict=True)
    if config_path.stat().st_size > 2_000_000:
        raise ValueError("Configuration exceeds 2 MB")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"), object_pairs_hook=json_object)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be an object")
    root = config_path.parent
    paths = {}
    for field in ("source", "before", "after"):
        paths[field] = local_path(root, config.get(field), field)
    output = local_path(root, config.get("output"), "output", output=True)
    source = read_image(paths["source"], "source")
    size = source.size
    images = {"source": source}
    for field in ("before", "after"):
        images[field] = read_image(paths[field], field, size=size)
    masks = {}
    if "keep" in config:
        paths["keep"] = local_path(root, config["keep"], "keep")
        masks["keep"] = read_image(paths["keep"], "keep", size=size, mask=True, binary=True)
    if ("alpha_before" in config) != ("alpha_after" in config):
        raise ValueError("alpha_before and alpha_after must be provided together")
    for field in ("alpha_before", "alpha_after"):
        if field in config:
            paths[field] = local_path(root, config[field], field)
            masks[field] = read_image(paths[field], field, size=size, mask=True)

    scales = config.get("scales_px")
    if (not isinstance(scales, list) or not 1 <= len(scales) <= 8
            or any(not finite_number(s) or not 0 < s <= min(4096, max(size)) for s in scales)
            or len(set(scales)) != len(scales)):
        raise ValueError("scales_px: require 1..8 unique finite positive radii, at most min(4096, canvas long edge)")
    regions = config.get("regions")
    if not isinstance(regions, list) or not 1 <= len(regions) <= MAX_REGIONS:
        raise ValueError(f"regions: require 1..{MAX_REGIONS} ROI records")
    seen = set()
    total_roi_pixels = 0
    for region in regions:
        if not isinstance(region, dict):
            raise ValueError("regions: each ROI must be an object")
        validate_id(region.get("id"), "regions", seen)
        box = region.get("box")
        if (not isinstance(box, list) or len(box) != 4
                or any(not isinstance(v, int) or isinstance(v, bool) for v in box)):
            raise ValueError("regions.box: require four integer coordinates")
        left, top, right, bottom = box
        if not (0 <= left < right <= size[0] and 0 <= top < bottom <= size[1]):
            raise ValueError(f"regions.{region['id']}: box lies outside the source canvas or is empty")
        total_roi_pixels += (right - left) * (bottom - top)
    # Contacts have three panes; error maps have two. Bound disk/memory demand.
    if total_roi_pixels * (10 + 4 * len(scales)) > MAX_OUTPUT_PIXELS:
        raise ValueError("regions: requested output image pixels exceed the bounded output budget")

    profiles = config.get("profiles", [])
    if not isinstance(profiles, list) or len(profiles) > MAX_PROFILES:
        raise ValueError(f"profiles: expected at most {MAX_PROFILES} records")
    seen = set()
    prepared_profiles = []
    total_samples = 0
    for profile in profiles:
        if not isinstance(profile, dict):
            raise ValueError("profiles: each record must be an object")
        validate_id(profile.get("id"), "profiles", seen)
        if profile.get("expected") not in ("continuous", "source_discontinuity"):
            raise ValueError("profiles.expected: require continuous or source_discontinuity")
        points = profile.get("points")
        if (not isinstance(points, list) or not 2 <= len(points) <= 256
                or any(not isinstance(p, list) or len(p) != 2
                       or any(not finite_number(v) for v in p) for p in points)):
            raise ValueError("profiles.points: require 2..256 finite [x,y] points")
        if any(not (0 <= x <= size[0] - 1 and 0 <= y <= size[1] - 1) for x, y in points):
            raise ValueError("profiles.points: pixel-center coordinates lie outside the source canvas")
        distance, coordinates = profile_coordinates(points)
        total_samples += len(distance)
        prepared_profiles.append((profile, distance, coordinates))
    if total_samples > MAX_PROFILE_SAMPLES:
        raise ValueError("profiles: total sample count exceeds the output limit")
    if any(p == output or p.is_relative_to(output) for p in [config_path, *paths.values()]):
        raise ValueError("output: must not overlap an input")
    hashes = {field: {"path": str(path.relative_to(root)), "sha256": sha256(path)}
              for field, path in paths.items()}
    return config, images, masks, output, prepared_profiles, {
        "configuration": {"path": config_path.name, "sha256": sha256(config_path)},
        "inputs": hashes,
    }


def linear_rgb(rgb):
    encoded = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4)


def luminance(rgb):
    return linear_rgb(rgb) @ LUMA_WEIGHTS


def difference_stats(source, candidate):
    residual = np.asarray(candidate, dtype=np.float32) - np.asarray(source, dtype=np.float32)
    error = np.abs(residual)
    return {
        "rgb_mae": float(error.mean()),
        "rgb_mae_per_channel": error.mean(axis=(0, 1)).tolist(),
        "rgb_max_abs_error": float(error.max()),
        "rgb_changed_pixels": int(np.any(error != 0, axis=2).sum()),
    }


def draw_font():
    # Pillow's own font is portable; all drawn labels intentionally use ASCII.
    try:
        return ImageFont.load_default(size=13)
    except TypeError:
        return ImageFont.load_default()


def contact(panes, labels, title):
    width, height = panes[0].size
    pane_width = max(width, 360)
    title_lines = textwrap.wrap(title, width=max(30, pane_width * len(panes) // 8))
    header = 34 + len(title_lines) * 18
    result = Image.new("RGB", (pane_width * len(panes), height + header), "white")
    draw = ImageDraw.Draw(result)
    font = draw_font()
    for index, line in enumerate(title_lines):
        draw.text((5, 3 + index * 18), line, fill="black", font=font)
    for index, (pane, label) in enumerate(zip(panes, labels)):
        left = index * pane_width
        result.paste(pane.convert("RGB"), (left, header))
        draw.text((left + 5, header - 24), label, fill="black", font=font)
    return result


def error_heat(error, gain=4.0):
    # Fixed scale across before/after/ROIs: white at >= 63.75 RGB codes.
    values = np.clip(np.mean(np.abs(error), axis=2) * gain, 0, 255).astype(np.uint8)
    return Image.fromarray(values, "L").convert("RGB")


def mask_checks(images, masks, output):
    result = {}
    if "keep" not in masks:
        result["keep"] = {"status": "unavailable", "reason": "keep mask not provided"}
    else:
        selected = masks["keep"] == 255
        source = np.asarray(images["source"])
        before_changed = selected & np.any(np.asarray(images["before"]) != source, axis=2)
        after_changed = selected & np.any(np.asarray(images["after"]) != source, axis=2)
        count = int(selected.sum())
        result["keep"] = {
            "status": "diagnostic" if count else "unavailable",
            "selected_pixels": count,
            "before_changed_pixels": int(before_changed.sum()) if count else None,
            "after_changed_pixels": int(after_changed.sum()) if count else None,
            "changed_map": "keep-changes.png",
            "map_legend": "red = before differs from source; green = after differs; yellow = both",
        }
        if not count:
            result["keep"]["reason"] = "keep mask contains no selected pixels"
        map_image = np.zeros((*selected.shape, 3), dtype=np.uint8)
        map_image[..., 0] = before_changed.astype(np.uint8) * 255
        map_image[..., 1] = after_changed.astype(np.uint8) * 255
        Image.fromarray(map_image, "RGB").save(output / "keep-changes.png")
    if "alpha_before" not in masks:
        result["alpha"] = {"status": "unavailable", "reason": "separate alpha masks not provided"}
    else:
        difference = np.abs(masks["alpha_after"].astype(np.int16) - masks["alpha_before"].astype(np.int16))
        result["alpha"] = {
            "status": "diagnostic", "changed_pixels": int((difference != 0).sum()),
            "max_abs_change": int(difference.max()), "changed_map": "alpha-changes.png",
        }
        Image.fromarray((difference != 0).astype(np.uint8) * 255, "L").save(output / "alpha-changes.png")
    return result


def region_report(region, images, output):
    identifier, box = region["id"], region["box"]
    roi_output = output / "regions" / identifier
    roi_output.mkdir(parents=True)
    crops = {key: image.crop(tuple(box)) for key, image in images.items()}
    source = np.asarray(crops["source"], dtype=np.float32)
    title = f"{identifier} | source ROI [{box[0]}, {box[1]}, {box[2]}, {box[3]}) | 1:1 pixels"
    contact(list(crops.values()), ["SOURCE", "BEFORE", "AFTER"], title).save(roi_output / "color.png")
    lumas = {key: luminance(image) for key, image in crops.items()}
    gray = [Image.fromarray(np.clip(value * 255, 0, 255).astype(np.uint8), "L") for value in lumas.values()]
    contact(gray, ["SOURCE", "BEFORE", "AFTER"], title + " | linear luminance * 255").save(roi_output / "gray.png")
    raw_errors = [np.asarray(crops[key], dtype=np.float32) - source for key in ("before", "after")]
    contact([error_heat(e) for e in raw_errors], ["BEFORE error", "AFTER error"],
            title + " | mean abs RGB error * 4; white >= 63.75").save(roi_output / "rgb-error.png")
    gray_errors = [Image.fromarray(np.clip(np.abs(lumas[key] - lumas["source"]) * 4 * 255, 0, 255).astype(np.uint8), "L")
                   for key in ("before", "after")]
    contact(gray_errors, ["BEFORE error", "AFTER error"],
            title + " | abs linear luminance error * 4; white >= 0.25").save(roi_output / "gray-error.png")
    report = {
        "id": identifier, "box": box, "status": "diagnostic",
        "rgb_units": RGB_UNIT, "luminance_units": LUMA_UNIT,
        "before": difference_stats(source, crops["before"]),
        "after": difference_stats(source, crops["after"]),
        "artifacts": {name: f"regions/{identifier}/{name}.png"
                      for name in ("color", "gray", "rgb-error", "gray-error")},
        "low_frequency": [],
    }
    for key in ("before", "after"):
        report[key]["linear_luminance_mae"] = float(np.mean(np.abs(lumas[key] - lumas["source"])))
    return report


def low_frequency_report(region, full_blurred_images, full_blurred_gray, index, scale, output):
    identifier, box = region["id"], region["box"]
    blurred = {key: image.crop(tuple(box)) for key, image in full_blurred_images.items()}
    blurred_source = np.asarray(blurred["source"], dtype=np.float32)
    errors = [np.asarray(blurred[key], dtype=np.float32) - blurred_source for key in ("before", "after")]
    filename = f"low-frequency-{index:02d}.png"
    title = f"{identifier} | source ROI {box} | Gaussian radius={scale:g}px; RGB error * 4"
    contact([error_heat(e) for e in errors], ["BEFORE low-frequency error", "AFTER low-frequency error"],
            title).save(output / "regions" / identifier / filename)
    gray = {key: np.asarray(image.crop(tuple(box)), dtype=np.float32) / 255.0
            for key, image in full_blurred_gray.items()}
    gray_filename = f"low-frequency-gray-{index:02d}.png"
    gray_errors = [Image.fromarray(np.clip(np.abs(gray[key] - gray["source"]) * 4 * 255, 0, 255).astype(np.uint8), "L")
                   for key in ("before", "after")]
    contact(gray_errors, ["BEFORE low-frequency luminance error", "AFTER low-frequency luminance error"],
            f"{identifier} | source ROI {box} | Gaussian radius={scale:g}px; abs luminance error * 4"
            ).save(output / "regions" / identifier / gray_filename)
    before_stats = difference_stats(blurred_source, blurred["before"])
    after_stats = difference_stats(blurred_source, blurred["after"])
    for key, stats in (("before", before_stats), ("after", after_stats)):
        stats["linear_luminance_mae_8bit"] = float(np.mean(np.abs(gray[key] - gray["source"])))
    return {
        "scale_px": scale,
        "method": "Pillow GaussianBlur on full-canvas encoded sRGB, then crop ROI; 8-bit diagnostic",
        "gray_method": "linear sRGB luminance quantized to 8-bit, full-canvas GaussianBlur, then crop; diagnostic",
        "before": before_stats,
        "after": after_stats,
        "artifact": f"regions/{identifier}/{filename}",
        "gray_artifact": f"regions/{identifier}/{gray_filename}",
    }


def bilinear(image, coordinates, *, linear_light=False):
    array = np.asarray(image)
    x, y = coordinates[:, 0], coordinates[:, 1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, array.shape[1] - 1), np.minimum(y0 + 1, array.shape[0] - 1)
    wx, wy = (x - x0)[:, None], (y - y0)[:, None]
    samples = [array[y0, x0], array[y0, x1], array[y1, x0], array[y1, x1]]
    if linear_light:
        samples = [linear_rgb(sample) for sample in samples]
    return ((1 - wx) * (1 - wy) * samples[0] + wx * (1 - wy) * samples[1]
            + (1 - wx) * wy * samples[2] + wx * wy * samples[3])


def residual_steps(source, candidate, distance, coordinates):
    residual = candidate - source
    delta = np.diff(residual, axis=0)
    step_distance = np.diff(distance)
    slope = delta / (step_distance[:, None] if delta.ndim == 2 else step_distance)
    score = np.max(np.abs(delta), axis=1) if delta.ndim == 2 else np.abs(delta)
    worst = int(np.argmax(score))
    return {
        "mean_abs_residual": np.mean(np.abs(residual), axis=0).tolist(),
        "mean_abs_residual_step": np.mean(np.abs(delta), axis=0).tolist(),
        "max_abs_residual_step": np.max(np.abs(delta), axis=0).tolist(),
        "mean_abs_residual_slope_per_px": np.mean(np.abs(slope), axis=0).tolist(),
        "max_abs_residual_slope_per_px": np.max(np.abs(slope), axis=0).tolist(),
        "worst_step": {
            "from_sample": worst, "to_sample": worst + 1,
            "distance_px": [float(distance[worst]), float(distance[worst + 1])],
            "source_coordinates": coordinates[worst:worst + 2].tolist(),
            "source_step": (source[worst + 1] - source[worst]).tolist(),
            "candidate_step": (candidate[worst + 1] - candidate[worst]).tolist(),
            "residual_step": delta[worst].tolist(),
        },
    }, delta


def plot_panel(draw, bounds, distance, series, label, font, *, fixed_range=None):
    left, top, right, bottom = bounds
    draw.text((left, top - 23), label, fill="black", font=font)
    if fixed_range is None:
        values = np.concatenate([np.asarray(values).ravel() for _, values, _, _ in series])
        low, high = min(0.0, float(values.min())), max(0.0, float(values.max()))
        if high - low < 1e-8:
            low, high = -1.0, 1.0
        else:
            padding = (high - low) * 0.05
            low, high = low - padding, high + padding
    else:
        low, high = fixed_range
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        y = bottom - fraction * (bottom - top)
        draw.line((left, y, right, y), fill="#dddddd", width=1)
        draw.text((4, y - 6), f"{low + fraction * (high - low):.4g}", fill="#444444", font=font)
    draw.rectangle(bounds, outline="#777777")
    xs = left + distance / max(float(distance[-1]), 1e-12) * (right - left)
    # Limit curve segments drawn, but CSV and metrics retain every sample.
    indices = np.unique(np.linspace(0, len(distance) - 1, min(len(distance), 3000)).astype(int))
    for index, (name, values, color, dashed) in enumerate(series):
        values = np.asarray(values)
        ys = bottom - (values - low) / (high - low) * (bottom - top)
        points = [(float(xs[i]), float(ys[i])) for i in indices]
        if dashed:
            for first, second in zip(points[:-1], points[1:]):
                # Distance-based pattern remains visible for sparse profiles.
                dx, dy = second[0] - first[0], second[1] - first[1]
                pieces = max(1, int(math.ceil(math.hypot(dx, dy) / 4)))
                for segment in range(pieces):
                    start = segment / pieces
                    finish = (segment + 1) / pieces
                    if int((first[0] + start * dx - left) / 5) % 2 == 0:
                        draw.line((first[0] + start * dx, first[1] + start * dy,
                                   first[0] + finish * dx, first[1] + finish * dy), fill=color, width=1)
        else:
            draw.line(points, fill=color, width=2)
        draw.text((left + 150 * index, bottom + 6), name, fill=color, font=font)
    draw.text((right - 190, bottom + 24), f"arc length 0..{distance[-1]:.3f} px", fill="black", font=font)


def profile_plot(path, profile, distance, values, lumas, residual, delta):
    chart = Image.new("RGB", (1200, 1230), "white")
    draw, font = ImageDraw.Draw(chart), draw_font()
    draw.text((12, 8), f"{profile['id']} | expected={profile['expected']} | DIAGNOSTIC ONLY", fill="black", font=font)
    draw.text((12, 30), "Source pixel centers: origin top-left; x right, y down. CSV gives x/y for every arc-length sample.", fill="black", font=font)
    draw.text((12, 52), "Residual = candidate - source. Before dashed; after solid. Real source jumps are retained in the reference.", fill="black", font=font)
    colors = {"source": "#202020", "before": "#c56000", "after": "#0069bb"}
    plot_panel(draw, (75, 115, 1175, 285), distance,
               [(key, lumas[key], colors[key], key == "before") for key in ("source", "before", "after")],
               "Linear sRGB relative luminance (0..1)", font, fixed_range=(0, 1))
    plot_panel(draw, (75, 395, 1175, 565), distance,
               [(key, lumas[key] - lumas["source"], colors[key], key == "before") for key in ("before", "after")],
               "Luminance residual to source", font)
    channel_colors = ("#bd2626", "#28742b", "#3155bc")
    rgb_series = [(f"{key} {channel}", residual[key][:, channel_index], channel_colors[channel_index], key == "before")
                  for key in ("before", "after") for channel_index, channel in enumerate("RGB")]
    plot_panel(draw, (75, 675, 1175, 845), distance, rgb_series,
               "RGB residual (encoded sRGB codes, -255..255); engineering values", font)
    delta_series = [(f"{key} {channel}", np.r_[0.0, delta[key][:, channel_index]], channel_colors[channel_index], key == "before")
                    for key in ("before", "after") for channel_index, channel in enumerate("RGB")]
    plot_panel(draw, (75, 955, 1175, 1125), distance, delta_series,
               "First difference of RGB residual per sampling step (sample 0 displayed as zero)", font)
    draw.text((12, 1188), "Plots may decimate long profiles; CSV and metrics retain all samples. Curves are evidence, never an automatic visual pass.",
              fill="black", font=font)
    chart.save(path)


def profile_report(prepared, images, output):
    profile, distance, coordinates = prepared
    identifier = profile["id"]
    directory = output / "profiles"
    directory.mkdir(exist_ok=True)
    values = {key: bilinear(image, coordinates) for key, image in images.items()}
    # Interpolate linear light before computing luminance to preserve its meaning.
    lumas = {key: bilinear(image, coordinates, linear_light=True) @ LUMA_WEIGHTS
             for key, image in images.items()}
    residual = {key: values[key] - values["source"] for key in ("before", "after")}
    result = {
        "id": identifier, "expected": profile["expected"], "status": "diagnostic",
        "points": profile["points"], "sample_count": len(distance),
        "arc_length_px": float(distance[-1]), "sample_step_px": float(distance[1] - distance[0]),
        "sampling": "uniform arc length <= 1 source pixel; bilinear, source pixel-center coordinates",
        "rgb_units": RGB_UNIT, "luminance_units": LUMA_UNIT,
        "residual_definition": "candidate minus source; residual step = candidate step minus source step",
        "expected_interpretation": "metadata for human review only; no smoothing target or pass threshold",
        "source_steps": {
            "max_abs_rgb_step": np.abs(np.diff(values["source"], axis=0)).max(axis=0).tolist(),
            "max_abs_luminance_step": float(np.abs(np.diff(lumas["source"])).max()),
        },
        "csv": f"profiles/{identifier}.csv", "plot": f"profiles/{identifier}.png",
    }
    rgb_delta, luma_delta = {}, {}
    for key in ("before", "after"):
        rgb_stats, rgb_delta[key] = residual_steps(values["source"], values[key], distance, coordinates)
        luma_stats, luma_delta[key] = residual_steps(lumas["source"], lumas[key], distance, coordinates)
        result[key] = {"rgb_residual": rgb_stats, "luminance_residual": luma_stats}
    header = ["sample", "arc_length_px", "source_x", "source_y"]
    for key in ("source", "before", "after"):
        header += [f"{key}_{channel}" for channel in ("r_code", "g_code", "b_code", "linear_luminance")]
    for key in ("before", "after"):
        header += [f"{key}_residual_{channel}" for channel in ("r_code", "g_code", "b_code", "linear_luminance")]
        header += [f"{key}_residual_step_{channel}" for channel in ("r_code", "g_code", "b_code", "linear_luminance")]
    with (directory / f"{identifier}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for index in range(len(distance)):
            row = [index, distance[index], *coordinates[index]]
            for key in ("source", "before", "after"):
                row += [*values[key][index], lumas[key][index]]
            for key in ("before", "after"):
                row += [*residual[key][index], lumas[key][index] - lumas["source"][index]]
                row += [*rgb_delta[key][index - 1], luma_delta[key][index - 1]] if index else ["", "", "", ""]
            writer.writerow(row)
    profile_plot(directory / f"{identifier}.png", profile, distance, values, lumas, residual, rgb_delta)
    return result


def run(config_path):
    config, images, masks, output, prepared_profiles, hashes = load_config(config_path)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "semantic-illustration-appearance-check/v1", "status": "diagnostic",
        "visual_pass": None,
        "limitations": [
            "No automatic visual pass. Inspect original-scale color/gray ROIs, isolated layers and black/white backgrounds.",
            "All composites must already share a background, registration and sRGB interpretation; this script cannot prove that.",
            "ROI MAE and low-frequency maps are engineering comparisons, not perceptual quality scores.",
            "Hidden content has no source truth; do not select occluding source objects as reconstruction ground truth.",
            "A zero residual step preserves the source step; true material/shadow discontinuities must not be flattened.",
        ],
        "canvas": list(images["source"].size), "hashes": hashes,
        "mask_checks": mask_checks(images, masks, output),
        "regions": [], "profiles": [],
        "profiles_status": "diagnostic" if prepared_profiles else "unavailable",
    }
    for region in config["regions"]:
        report["regions"].append(region_report(region, images, output))
    # Filter each full canvas once per scale, then crop; ROI borders do not invent seams.
    gray_images = {key: Image.fromarray(np.clip(luminance(image) * 255, 0, 255).astype(np.uint8), "L")
                   for key, image in images.items()}
    for index, scale in enumerate(config["scales_px"]):
        blurred_images = {key: image.filter(ImageFilter.GaussianBlur(float(scale)))
                          for key, image in images.items()}
        blurred_gray = {key: image.filter(ImageFilter.GaussianBlur(float(scale)))
                        for key, image in gray_images.items()}
        for region, region_result in zip(config["regions"], report["regions"]):
            region_result["low_frequency"].append(low_frequency_report(region, blurred_images, blurred_gray, index, scale, output))
    del gray_images, blurred_images, blurred_gray
    profile_arrays = {key: np.asarray(image) for key, image in images.items()} if prepared_profiles else {}
    for prepared in prepared_profiles:
        report["profiles"].append(profile_report(prepared, profile_arrays, output))
    report_path = output / "appearance-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Project-local appearance-review JSON; output directory must not exist")
    args = parser.parse_args(argv)
    try:
        report_path = run(args.config)
    except (ValueError, OSError, Image.DecompressionBombError) as error:
        print(json.dumps({"status": "error", "message": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"status": "diagnostic", "report": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
