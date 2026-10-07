# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 ChitoseRen
"""Compare aligned illustration composites and count explicitly reviewed semantic parts.

QA only. Dependencies: numpy, Pillow. Never changes artistic input pixels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CRITERIA = {
    "object_semantics": "对象语义完整，分区及归属正确",
    "hidden_completion": "会暴露的隐藏结构与连接已补齐且合理",
    "no_foreign_pixels": "没有其他前景部件残留、重影或旧轮廓",
    "alpha_edges": "透明孔隙、轮廓、抗锯齿及边缘质量合格",
    "static_assembly": "默认静态拼合位置、层序及原画表现合格",
    "required_motion": "已在要求的运动/表情范围内检查且无缺口",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rgb(path):
    im = Image.open(path)
    if "A" in im.getbands() and im.getchannel("A").getextrema()[0] < 255:
        raise ValueError(f"{path}: source/candidate must be opaque full composites; flatten over the intended background first. Pass foreground alpha separately.")
    return im.convert("RGB")


def load_mask(path, size):
    im = Image.open(path)
    if im.size != size:
        raise ValueError(f"Mask canvas mismatch: {path}")
    # RGBA means use alpha, grayscale means use pixel value. Never infer subject
    # from the RGB brightness of an opaque colour image.
    if "A" in im.getbands():
        a = im.getchannel("A")
    elif im.mode in ("1", "L", "I", "I;16", "F"):
        a = im.convert("L")
    else:
        array = np.asarray(im.convert("RGB"))
        if not np.array_equal(array[..., 0], array[..., 1]) or not np.array_equal(array[..., 0], array[..., 2]):
            raise ValueError(f"{path}: use a grayscale mask or RGBA alpha, not an opaque colour illustration")
        a = im.convert("L")
    return np.asarray(a, dtype=np.float64) / 255


def mae_stats(error, region):
    pixels = int(region.sum())
    if pixels == 0:
        return {"pixels": 0, "normalized_rgb_mae": None, "percent_of_rgb_range": None}
    values = error[region]
    pixel_mean = values.mean(axis=-1)
    return {
        "pixels": pixels,
        "normalized_rgb_mae": float(values.mean()),
        "percent_of_rgb_range": float(values.mean() * 100),
        "per_channel_normalized_mae": values.mean(axis=0).tolist(),
        "pixel_mean_error_p50": float(np.percentile(pixel_mean, 50)),
        "pixel_mean_error_p95": float(np.percentile(pixel_mean, 95)),
        "fraction_pixels_mean_error_over_0_15": float((pixel_mean > .15).mean()),
        "fraction_pixels_any_channel_error_over_0_15": float((values.max(axis=-1) > .15).mean()),
    }


def box_mean(array, window=11):
    pad = window // 2
    a = np.pad(array, ((pad, pad), (pad, pad)), mode="reflect")
    integral = np.pad(a, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    return (integral[window:, window:] - integral[:-window, window:]
            - integral[window:, :-window] + integral[:-window, :-window]) / window**2


def ssim_map(source, candidate, window=11):
    # Population covariance, uniform 11x11 window, reflected borders.
    x = source @ np.array([.2126, .7152, .0722])
    y = candidate @ np.array([.2126, .7152, .0722])
    ux, uy = box_mean(x, window), box_mean(y, window)
    vx = np.maximum(box_mean(x*x, window) - ux*ux, 0)
    vy = np.maximum(box_mean(y*y, window) - uy*uy, 0)
    vxy = box_mean(x*y, window) - ux*uy
    return ((2*ux*uy + .01**2) * (2*vxy + .03**2) /
            ((ux*ux + uy*uy + .01**2) * (vx + vy + .03**2)))


def metrics(source, candidate, source_alpha, candidate_alpha=None):
    error = np.abs(source - candidate)
    source_fg = source_alpha >= .5
    if candidate_alpha is None:
        union = source_fg
        silhouette = {"available": False, "reason": "candidate subject alpha not supplied; silhouette cannot be inferred from the opaque composite"}
        roi_kind = "source_foreground_only_candidate_alpha_not_supplied"
    else:
        candidate_fg = candidate_alpha >= .5
        union = source_fg | candidate_fg
        intersection = source_fg & candidate_fg
        source_area, candidate_area = int(source_fg.sum()), int(candidate_fg.sum())
        silhouette = {
            "available": True, "binary_alpha_threshold": .5,
            "intersection_over_union": float(intersection.sum() / union.sum()) if union.any() else None,
            "source_area_pixels": source_area, "candidate_area_pixels": candidate_area,
            "signed_area_difference_pixels": candidate_area - source_area,
            "signed_area_difference_fraction_of_source": (candidate_area-source_area)/source_area if source_area else None,
            "absolute_area_difference_fraction_of_source": abs(candidate_area-source_area)/source_area if source_area else None,
            "symmetric_difference_pixels": int((source_fg ^ candidate_fg).sum()),
            "soft_alpha_iou": float(np.minimum(source_alpha, candidate_alpha).sum() / np.maximum(source_alpha, candidate_alpha).sum()) if np.maximum(source_alpha, candidate_alpha).sum() else None,
            "source_soft_alpha_area": float(source_alpha.sum()),
            "candidate_soft_alpha_area": float(candidate_alpha.sum()),
            "note": "Equal area alone does not imply equal silhouette; inspect IoU and symmetric difference.",
        }
        roi_kind = "union_of_source_and_candidate_subject_masks"
    regions = {"full_image": np.ones(source_fg.shape, dtype=bool),
               "foreground_union_roi": union, "source_foreground_roi": source_fg,
               "background_outside_union": ~union}
    colour = {name: mae_stats(error, region) for name, region in regions.items()}
    structure = ssim_map(source, candidate)
    return {
        "colour": colour,
        "foreground_roi_kind": roi_kind,
        "foreground_union_fraction_of_canvas": float(union.mean()),
        "silhouette": silhouette,
        "ssim": {
            "implementation": "numpy luminance SSIM; 11x11 uniform window; population covariance; reflected borders; K1=.01,K2=.03,data_range=1",
            "luminance_weights_on_encoded_RGB": [.2126, .7152, .0722],
            "note": "A reproducible structural indicator, not a perceptual percent difference or identical to every library's Gaussian-window SSIM.",
            **{name: float(structure[region].mean()) if region.any() else None for name, region in regions.items()},
        },
    }, error.mean(axis=-1), union


def read_parts(path):
    manifest_root = Path(path).resolve().parent
    items = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(items, dict):
        items = items["parts"]
        if isinstance(items, dict):
            items = list(items.values())
    if len({m["id"] for m in items}) != len(items):
        raise ValueError("Manifest contains duplicate drawing IDs")
    # Geometry and motion keep unique drawing IDs. Acceptance groups all
    # expected drawings, including missing/default-hidden segments, by logical
    # ID so splitting an interleaved object cannot inflate the denominator.
    groups = {}
    for item in items:
        item = dict(item)
        # Explicit null means unavailable, never a fallback to a stale output.
        value = item["file"] if "file" in item else item.get("path", "parts/" + item["id"] + ".png")
        item["resolved_asset_path"] = str((manifest_root / value).resolve()) if value else None
        logical_id = item.get("logical_id", item["id"])
        if not isinstance(logical_id, str) or not logical_id.strip():
            raise ValueError("logical_id must be a nonempty string")
        group = groups.setdefault(logical_id, {"id": logical_id, "drawings": []})
        group["drawings"].append(item)
    for group in groups.values():
        drawings = group["drawings"]
        names = list(dict.fromkeys(d.get("name", d["id"]) for d in drawings))
        group["name"] = " / ".join(names)
        group["drawing_ids"] = [d["id"] for d in drawings]
        if len(drawings) == 1 and "number" in drawings[0]:
            group["number"] = drawings[0]["number"]
    return list(groups.values())


def write_score_template(path, parts):
    # Preserve an existing manually edited score sheet.
    if path.exists():
        return False
    fields = ["number", "id", "name", "drawing_ids", *CRITERIA, "reviewer", "reviewed_at", "evidence", "notes", "asset_sha256"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for i, m in enumerate(parts):
            writer.writerow({"number": m.get("number", i+1), "id": m["id"], "name": m.get("name", ""),
                             "drawing_ids": ";".join(m.get("drawing_ids", [m["id"]])),
                             **{criterion: "pending" for criterion in CRITERIA}})
    return True


def count_scores(path, parts):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    expected, actual = {m["id"] for m in parts}, [r["id"] for r in rows]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Score sheet must contain every expected semantic part exactly once; extras/subsets/duplicates are not accepted")
    counts = {"passed": 0, "failed": 0, "pending": 0}
    by_id = {part["id"]: part for part in parts}
    results = []
    for row in rows:
        values = {key: row.get(key, "").strip().lower() or "pending" for key in CRITERIA}
        if any(value not in ("pass", "fail", "pending") for value in values.values()):
            raise ValueError("Scores must be pass/fail/pending; record non-applicability rationale in notes, not an automatic pass: " + row["id"])
        status = "failed" if "fail" in values.values() else "passed" if all(v == "pass" for v in values.values()) else "pending"
        drawing_ids = by_id[row["id"]].get("drawing_ids", [row["id"]])
        evidence = row.get("evidence", "").strip()
        missing_drawings = drawing_ids[:] if not evidence else []
        evidence_error = ""
        if len(drawing_ids) > 1:
            # A free-text global contact reference cannot identify which
            # segments were reviewed. Require explicit per-drawing evidence.
            try:
                by_drawing = json.loads(evidence)
                if not isinstance(by_drawing, dict):
                    raise ValueError("evidence must be a JSON object")
                missing_drawings = [pid for pid in drawing_ids
                                    if not isinstance(by_drawing.get(pid), str) or not by_drawing[pid].strip()]
                unknown = sorted(set(by_drawing) - set(drawing_ids))
                if unknown:
                    evidence_error = "Evidence contains unknown drawing IDs: " + ", ".join(unknown)
            except (ValueError, TypeError):
                missing_drawings = drawing_ids[:]
                evidence_error = "Multi-drawing evidence must be a JSON object mapping each drawing id to its nonempty file/hash and review evidence"
        # Passing without evidence for every drawing is left unverified.
        evidence_missing = bool(missing_drawings or evidence_error)
        if evidence_missing:
            if status == "passed":
                status = "pending"
        # Bind the recorded review to CURRENT manifest files, never CSV paths
        # or a newly calculated hash silently written into the review record.
        raw_hashes = row.get("asset_sha256", "").strip()
        recorded_hashes = {drawing_ids[0]: raw_hashes} if len(drawing_ids) == 1 else {}
        hash_error = ""
        if len(drawing_ids) > 1:
            try:
                recorded_hashes = json.loads(raw_hashes)
                if not isinstance(recorded_hashes, dict):
                    raise ValueError("asset_sha256 must be a JSON object")
                unknown_hash_ids = sorted(set(recorded_hashes) - set(drawing_ids))
                if unknown_hash_ids:
                    hash_error = "Hash map contains unknown drawing IDs: " + ", ".join(unknown_hash_ids)
            except (ValueError, TypeError):
                recorded_hashes = {}
                hash_error = "Multi-drawing asset_sha256 must map each drawing id to its recorded SHA256"
        bindings = []
        for drawing in by_id[row["id"]].get("drawings", [{"id": row["id"]}]):
            pid = drawing["id"]
            recorded = recorded_hashes.get(pid)
            filename = drawing.get("resolved_asset_path")
            current = digest(filename) if filename and Path(filename).is_file() else None
            valid = isinstance(recorded, str) and re.fullmatch(r"[0-9a-fA-F]{64}", recorded) is not None
            reason = ("missing_or_invalid_recorded_hash" if not valid else
                      "current_drawing_file_missing" if current is None else
                      "hash_mismatch" if current != recorded.lower() else "")
            bindings.append({"id": pid, "path": filename, "recorded_sha256": recorded,
                             "current_sha256": current, "reason": reason})
        stale_drawings = [binding["id"] for binding in bindings if binding["reason"]]
        if stale_drawings or hash_error:
            status = "pending"
        counts[status] += 1
        results.append({"id": row["id"], "status": status, "criteria": values,
                        "drawing_ids": drawing_ids,
                        "evidence_missing": evidence_missing,
                        "evidence_missing_drawing_ids": missing_drawings,
                        "evidence_error": evidence_error, "evidence": evidence,
                        "stale_drawing_ids": stale_drawings,
                        "hash_error": hash_error, "drawing_hash_bindings": bindings,
                        "reviewer": row.get("reviewer", ""), "notes": row.get("notes", "")})
    n = len(parts)
    if not n:
        raise ValueError("No expected semantic parts")
    return {
        "definition": "logical parts with all six criteria explicitly pass, evidence covering every drawing and matching recorded/current drawing hashes / all expected logical parts; stale or pending parts remain in denominator",
        "criteria": CRITERIA, "counts": counts, "expected_parts": n,
        "expected_drawings": sum(len(part.get("drawing_ids", [part["id"]])) for part in parts),
        "verified_success_rate": counts["passed"]/n,
        "verified_success_percent": counts["passed"]*100/n,
        "not_an_automatic_visual_judgment": True,
        "parts": results,
    }


def get_font(size=18):
    windows_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    candidates = (
        windows_fonts / "msyh.ttc", windows_fonts / "simhei.ttf",
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        "NotoSansCJK-Regular.ttc", "DejaVuSans.ttf",
    )
    for name in candidates:
        try:
            return ImageFont.truetype(str(name), size)
        except OSError:
            pass
    return ImageFont.load_default()


def contact(images, labels, target):
    tile_width, tile_height = 410, 780
    out = Image.new("RGB", (tile_width*len(images), tile_height), (36, 40, 48))
    d = ImageDraw.Draw(out)
    for i, (im, label) in enumerate(zip(images, labels)):
        frame = im.copy()
        frame.thumbnail((tile_width-10, tile_height-56), Image.Resampling.LANCZOS)
        out.paste(frame, (i*tile_width+(tile_width-frame.width)//2, 50))
        d.text((i*tile_width+8, 10), label, fill="white", font=get_font(17))
    out.save(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source-mask", type=Path, required=True)
    parser.add_argument("--candidate-alpha", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.scores and not args.manifest:
        parser.error("--scores requires --manifest so the complete denominator is known")
    source_im, candidate_im = load_rgb(args.source), load_rgb(args.candidate)
    if source_im.size != candidate_im.size:
        raise ValueError("Canvas mismatch. Align composites explicitly; no silent resize is allowed")
    source_alpha = load_mask(args.source_mask, source_im.size)
    candidate_alpha = load_mask(args.candidate_alpha, source_im.size) if args.candidate_alpha else None
    source = np.asarray(source_im, dtype=np.float64)/255
    candidate = np.asarray(candidate_im, dtype=np.float64)/255
    report, error, union = metrics(source, candidate, source_alpha, candidate_alpha)
    report["canvas"] = source_im.size
    report["source_encoding"] = "8-bit encoded RGB values; no hidden alignment, rescaling or ICC conversion. Inputs should share the same intended colour space."
    report["artistic_inputs_modified"] = False
    report["inputs"] = {key: {"path": str(path.resolve()), "sha256": digest(path)} for key, path in
                        [("source", args.source), ("candidate", args.candidate), ("source_mask", args.source_mask),
                         ("candidate_alpha", args.candidate_alpha)] if path is not None}
    args.out.mkdir(parents=True, exist_ok=True)
    # Fixed heat scale makes reports comparable: black=0, yellow~=7.5%, red>=15%.
    strength = np.clip(error/.15, 0, 1)
    heat = np.stack((np.minimum(strength*2, 1), np.minimum(strength*2, 2-strength*2), np.zeros_like(strength)), axis=-1)
    heat_im = Image.fromarray(np.round(heat*255).astype(np.uint8), "RGB")
    heat_im.save(args.out / "difference-heatmap.png")
    Image.fromarray(np.round(np.clip(error, 0, 1)*255).astype(np.uint8), "L").save(args.out / "difference-mean-rgb.png")
    # Explicit masks on the candidate make the foreground/background denominator visible.
    overlay = np.asarray(candidate_im).copy().astype(float)
    overlay[union] = overlay[union]*.7 + np.array([0, 210, 220])*.3
    roi_im = Image.fromarray(np.round(overlay).astype(np.uint8), "RGB")
    roi_im.save(args.out / "foreground-union-overlay.png")
    contact([source_im, candidate_im, heat_im, roi_im],
            ["原图", "候选复合", "RGB均差热图：红色≥15%", "青色：主体评估ROI"], args.out / "fidelity-contact.png")
    if candidate_alpha is not None:
        s, c = source_alpha >= .5, candidate_alpha >= .5
        silhouette = np.zeros((*s.shape, 3), dtype=np.uint8)
        silhouette[s & c] = [190, 190, 190]
        silhouette[s & ~c] = [255, 40, 80]
        silhouette[c & ~s] = [0, 210, 255]
        Image.fromarray(silhouette).save(args.out / "silhouette-difference.png")
        report["silhouette"]["difference_visual_legend"] = "grey=intersection; red=source subject missing from candidate; cyan=candidate subject outside source"
    if args.manifest:
        parts = read_parts(args.manifest)
        template = args.out / "semantic-part-scores.csv"
        write_score_template(template, parts)
        report["semantic_part_review"] = count_scores(args.scores or template, parts)
        report["semantic_part_review"]["scores_path"] = str((args.scores or template).resolve())
    else:
        report["semantic_part_review"] = {"available": False, "reason": "manifest not supplied; no semantic success rate can be computed"}
    (args.out / "fidelity-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "criteria.json").write_text(json.dumps(CRITERIA, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str((args.out / "fidelity-report.json").resolve()),
                      "global_RGB_MAE_percent": report["colour"]["full_image"]["percent_of_rgb_range"],
                      "foreground_ROI_RGB_MAE_percent": report["colour"]["foreground_union_roi"]["percent_of_rgb_range"],
                      "user_visual_target_verified": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
