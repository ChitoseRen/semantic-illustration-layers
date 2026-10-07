# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 ChitoseRen
"""Portable geometry-only preparation, RGBA registration and layer composition.

Usage: python asset_pipeline.py {prepare,register,compose} project.json
Dependencies: Pillow, numpy. No segmentation, painting, inpainting, or image generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from relations import state_visibility, validate_relations


PART_METADATA_FIELDS = ("logical_id", "baseline_unit_id", "semantic_parent", "motion_parent",
                        "depends_on", "semantic_requirements", "hidden_requirements", "line_owner",
                        "connection_points", "coordinate_space", "material_id", "paint_role",
                        "receiver_id", "caster_ids", "caster_note", "motion_policy")


def part_metadata(part):
    metadata = {key: part[key] for key in PART_METADATA_FIELDS if key in part}
    # File readiness is independent of evidence that an object is positioned
    # correctly. Do not propagate a caller-authored visual pass through geometry.
    metadata["position_validation"] = "unverified"
    return metadata


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def relative(root, path):
    return Path(path).resolve().relative_to(root).as_posix()


def resolve(root, value, output=False):
    path = Path(value)
    if path.is_absolute():
        raise ValueError("Manifest paths must be relative to project.json: " + str(value))
    resolved = (root / path).resolve()
    if output and not resolved.is_relative_to(root):
        raise ValueError("Outputs must remain inside the project directory")
    return resolved


def load_project(filename):
    filename = Path(filename).resolve()
    root = filename.parent
    project = json.loads(filename.read_text(encoding="utf-8-sig"))
    source = resolve(root, project["source"])
    with Image.open(source) as image:
        size = image.size
    parts = project["parts"]
    ids = [part["id"] for part in parts]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("parts must contain unique drawing IDs and cannot be empty")
    if any(not isinstance(pid, str) or not re.fullmatch(r"[\w-]+", pid) for pid in ids):
        raise ValueError("Part IDs may contain Unicode letters, numbers, underscores and hyphens")
    if any(not isinstance(part.get("logical_id", part["id"]), str)
           or not re.fullmatch(r"[\w-]+", part.get("logical_id", part["id"])) for part in parts):
        raise ValueError("logical_id must be a nonempty stable ID; omit it to use the drawing id")
    validate_relations(project, size)
    out = resolve(root, project.get("output", "output"), output=True)
    inputs = {source}
    for value in [project.get("background"), *(part.get("asset") for part in parts)]:
        if value:
            inputs.add(resolve(root, value))
    targets = [out / "parts" / (pid + ".png") for pid in ids]
    targets += [out / "inputs" / (pid + ".png") for pid in ids]
    targets += [out / name for name in ("foreground.png", "composite.png", "foreground.incomplete.png", "composite.incomplete.png")]
    for state in project.get("states", []):
        sid = state["id"]
        if not re.fullmatch(r"[\w-]+", sid):
            raise ValueError("State IDs must contain letters/numbers, underscores or hyphens for safe output filenames")
        if sid != "default":
            targets += [out / f"{stem}.{sid}{suffix}.png" for stem in ("foreground", "composite")
                        for suffix in ("", ".incomplete")]
    if any(target.resolve() in inputs for target in targets):
        raise ValueError("An output would overwrite an artistic input; choose another output directory")
    return project, root, source, size, out


def prepare(filename):
    project, root, source, size, out = load_project(filename)
    image = Image.open(source)
    records = []
    for part in project["parts"]:
        crop = part.get("crop", [0, 0, *size])
        if len(crop) != 4 or any(int(x) != x for x in crop):
            raise ValueError("crop must contain four integer source coordinates")
        x0, y0, x1, y1 = map(int, crop)
        if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
            raise ValueError("crop outside source canvas for " + part["id"])
        target = out / "inputs" / (part["id"] + ".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        image.crop((x0, y0, x1, y1)).save(target)
        records.append({"id": part["id"],
                        **part_metadata(part),
                        "input": relative(root, target), "crop": crop,
                        "input_size": [x1-x0, y1-y0], "source": project["source"],
                        "purpose": "reference crop only; not a completed semantic asset"})
    report = {"status": "prepared", "canvas": size, "source_sha256": digest(source), "parts": records}
    save_json(out / "prepared.json", report)
    return report


def fit_landmarks(spec):
    points = spec["landmarks"]
    src, dst = np.asarray(points["asset"], float), np.asarray(points["canvas"], float)
    if src.ndim != 2 or src.shape[1] != 2 or src.shape != dst.shape or not np.isfinite([src, dst]).all():
        raise ValueError("landmarks.asset and landmarks.canvas must be matching finite Nx2 arrays")
    method = spec.get("method", "similarity")
    if method == "affine":
        design = np.c_[src, np.ones(len(src))]
        if len(src) < 3 or np.linalg.matrix_rank(design) < 3:
            raise ValueError("Affine needs at least three non-collinear asset landmarks")
        matrix = np.linalg.lstsq(design, dst, rcond=None)[0].T
    elif method == "similarity":
        if len(src) < 2 or np.linalg.norm(src-src[0], axis=1).max() < 1e-8:
            raise ValueError("Similarity needs at least two distinct asset landmarks")
        candidates = []
        for sign in ([1, -1] if spec.get("allow_reflection", False) else [1]):
            design, target = [], []
            for (x, y), (u, v) in zip(src, dst):
                design.extend([[x, -sign*y, 1, 0], [sign*y, x, 0, 1]])
                target.extend([u, v])
            a, b, tx, ty = np.linalg.lstsq(design, target, rcond=None)[0]
            candidate = np.array([[a, -sign*b, tx], [b, sign*a, ty]])
            loss = np.sum((np.c_[src, np.ones(len(src))] @ candidate.T - dst)**2)
            candidates.append((loss, candidate))
        matrix = min(candidates, key=lambda c: c[0])[1]
    else:
        raise ValueError("Landmark method must be similarity or affine")
    errors = np.linalg.norm(np.c_[src, np.ones(len(src))] @ matrix.T - dst, axis=1)
    return matrix, errors.tolist()


def registration_matrix(spec, asset_size, canvas_size):
    if "frame" in spec and "landmarks" in spec:
        raise ValueError("Choose frame or landmarks, not both")
    if "landmarks" in spec:
        matrix, errors = fit_landmarks(spec)
    elif "frame" in spec:
        x0, y0, x1, y1 = spec["frame"]
        matrix = np.array([[(x1-x0)/asset_size[0], 0, x0], [0, (y1-y0)/asset_size[1], y0]], float)
        errors = []
    elif asset_size == canvas_size:
        matrix, errors = np.array([[1., 0, 0], [0, 1., 0]]), []
    else:
        raise ValueError("Different-sized assets require explicit frame or landmarks")
    determinant = float(np.linalg.det(matrix[:, :2]))
    if not np.isfinite(matrix).all() or abs(determinant) < 1e-10:
        raise ValueError("Registration is singular or non-finite")
    if determinant < 0 and spec.get("allow_reflection") is not True:
        raise ValueError("Negative determinant requires explicit allow_reflection: true")
    return matrix, errors, determinant


def warp_rgba(image, matrix, size):
    array = np.asarray(image.convert("RGBA"), np.float32)/255
    array[..., :3] *= array[..., 3:4]
    # A direct affine BICUBIC sample does not low-pass large reductions: thin
    # outlines can disappear or become dotted. Prefilter premultiplied channels
    # before the inverse warp; retain the declared canvas transform exactly.
    scale = min(1., float(np.linalg.svd(matrix[:, :2], compute_uv=False).min()))
    if scale < .98:
        width, height = image.size
        reduced = (max(1, round(width*scale)), max(1, round(height*scale)))
        array = np.stack([np.asarray(Image.fromarray(array[..., c], "F").resize(
            reduced, Image.Resampling.LANCZOS), np.float32) for c in range(4)], -1)
        matrix = matrix @ np.diag([width/reduced[0], height/reduced[1], 1.])
    inverse = np.linalg.inv(np.vstack([matrix, [0, 0, 1]]))[:2].ravel()
    channels = [np.asarray(Image.fromarray(array[..., channel], "F").transform(
        size, Image.Transform.AFFINE, tuple(inverse), Image.Resampling.BICUBIC), np.float32)
        for channel in range(4)]
    result = np.clip(np.stack(channels, -1), 0, 1)
    result[..., :3] = np.divide(result[..., :3], result[..., 3:4],
                               out=np.zeros_like(result[..., :3]), where=result[..., 3:4] > 1e-6)
    return Image.fromarray(np.uint8(np.clip(result*255+.5, 0, 255)), "RGBA")


def register_image(image, part, size):
    """One registration path for production and stale-asset checks."""
    space = part.get("coordinate_space", "asset")
    if space not in ("source_canvas", "asset"):
        raise ValueError("coordinate_space must be source_canvas or asset")
    if "A" not in image.getbands() and "transparency" not in image.info:
        raise ValueError("Asset must carry an explicit alpha channel or palette transparency")
    if space == "source_canvas" and image.size != size:
        raise ValueError("source_canvas assets must match the source pixel dimensions")
    matrix, errors, determinant = registration_matrix(part.get("registration", {}), image.size, size)
    resampling = "premultiplied LANCZOS minification prefilter then inverse BICUBIC affine"
    if space == "source_canvas":
        identity = np.array([[1., 0, 0], [0, 1., 0]])
        if not np.allclose(matrix, identity, atol=1e-9, rtol=0):
            raise ValueError("source_canvas assets require identity registration; remove the previous candidate transform")
        matrix, determinant = identity, 1.0
        registered = image.convert("RGBA").copy()
        resampling = "none: declared source_canvas with identity registration"
    else:
        registered = warp_rgba(image, matrix, size)
    return registered, matrix, errors, determinant, resampling


def register(filename):
    project, root, source, size, out = load_project(filename)
    visible = state_visibility(project)
    records = []
    for index, part in enumerate(project["parts"]):
        record = {"id": part["id"], "name": part.get("name", part["id"]),
                  **part_metadata(part),
                  "order": part.get("order", index), "visible": part["id"] in visible}
        if not part.get("asset") or not resolve(root, part["asset"]).is_file():
            records.append({**record, "status": "missing", "reason": "No existing RGBA asset; no source fallback is used"})
            continue
        try:
            asset = resolve(root, part["asset"])
            with Image.open(asset) as opened:
                image = opened.copy()
            registered, matrix, errors, determinant, resampling = register_image(image, part, size)
            target = out / "parts" / (part["id"] + ".png")
            target.parent.mkdir(parents=True, exist_ok=True)
            registered.save(target)
            record.update({"status": "ready", "file": relative(root, target), "asset": part["asset"],
                           "asset_sha256": digest(asset), "asset_size": image.size,
                           "resampling": resampling,
                           "transform_asset_to_canvas": matrix.tolist(), "determinant": determinant,
                           "reflection_explicitly_allowed": part.get("registration", {}).get("allow_reflection") is True,
                           "landmark_errors_px": errors, "alpha_bbox": registered.getchannel("A").getbbox()})
            if record["alpha_bbox"] is None:
                record.update(status="invalid", reason="Registered asset has no visible pixels on the canvas")
        except (ValueError, OSError, KeyError, TypeError, np.linalg.LinAlgError) as error:
            record.update(status="invalid", reason=str(error))
        records.append(record)
    report = {"status": "complete" if all(r["status"] == "ready" for r in records) else "incomplete",
              "canvas": size, "source_sha256": digest(source), "parts": records,
              "relations": validate_relations(project, size),
              "note": "Geometric registration only. Residual and complete status do not imply visual or motion acceptance."}
    save_json(out / "registration.json", report)
    return report


def compose(filename, state_id="default"):
    # Re-register declared inputs each time; never consume a stale saved layer
    # for an asset removed from the current project manifest.
    project, root, source, size, out = load_project(filename)
    if not re.fullmatch(r"[\w-]+", state_id):
        raise ValueError("State ID used for output filenames must contain letters/numbers, underscores or hyphens")
    if "alignment_review" in project:
        settings = project["alignment_review"]
        if not isinstance(settings, dict) or not isinstance(settings.get("regions"), list) or not settings["regions"]:
            raise ValueError("alignment_review configuration requires a nonempty regions list")
    visible = state_visibility(project, state_id)
    report = register(filename)
    for record in report["parts"]:
        record["visible"] = record["id"] in visible
    foreground = Image.new("RGBA", size)
    ordered = sorted(report["parts"], key=lambda r: r["order"])
    for record in ordered:
        if record["status"] == "ready" and record["visible"]:
            foreground.alpha_composite(Image.open(resolve(root, record["file"])).convert("RGBA"))
    suffix = ("" if state_id == "default" else "." + state_id)
    suffix += "" if report["status"] == "complete" else ".incomplete"
    foreground_path = out / ("foreground" + suffix + ".png")
    foreground.save(foreground_path)
    composite_path = None
    if project.get("background"):
        background = Image.open(resolve(root, project["background"])).convert("RGBA")
        if background.size != size:
            raise ValueError("Background must match source canvas; align it explicitly")
        background.alpha_composite(foreground)
        composite_path = out / ("composite" + suffix + ".png")
        background.save(composite_path)
    manifest = [{"id": part["id"], "name": part["name"], "order": part["order"],
                 **part_metadata(part),
                 "file": "parts/" + part["id"] + ".png" if part["status"] == "ready" else None,
                 "status": part["status"],
                 "visible": part["visible"]} for part in report["parts"]]
    state_suffix = "" if state_id == "default" else "." + state_id
    save_json(out / ("full-canvas-manifest" + state_suffix + ".json"), manifest)
    report.update(foreground=relative(root, foreground_path),
                  composite=relative(root, composite_path) if composite_path else None,
                  background_status="provided" if composite_path else "not_provided_no_composite_created",
                  state_id=state_id,
                  order_back_to_front=[r["id"] for r in ordered if r["visible"]],
                  missing_or_invalid_parts=[r["id"] for r in ordered if r["status"] != "ready"])
    if "alignment_review" in project:
        # Only the actual current-state assembly is reviewed. A transparent
        # foreground is passed as-is; missing pixels are never filled from source.
        from alignment_review import create_review
        review_out = out / "alignment" / state_id / uuid.uuid4().hex
        if not review_out.resolve().is_relative_to(root):
            raise ValueError("alignment_review output leaves the project directory")
        report["alignment_review"] = create_review(source, composite_path or foreground_path,
                                                     project["alignment_review"], root, review_out)
    save_json(out / ("composition" + state_suffix + ".json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "register", "compose"))
    parser.add_argument("project", type=Path)
    parser.add_argument("--state", default="default", help="Named visibility state for compose")
    args = parser.parse_args()
    try:
        if args.state != "default" and args.command != "compose":
            parser.error("--state is supported by compose only")
        report = (compose(args.project, args.state) if args.command == "compose"
                  else {"prepare": prepare, "register": register}[args.command](args.project))
        print(json.dumps(report, ensure_ascii=False))
        return 2 if report["status"] == "incomplete" else 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
