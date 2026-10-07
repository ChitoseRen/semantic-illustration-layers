# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 ChitoseRen
"""Non-destructive, layer-by-layer material-motion QA. This is not a delivered rig.

All drawing/warping here produces diagnostic previews only; no source parts change.
Dependencies: Python 3.10+, Pillow, numpy.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from fidelity_report import get_font as font

VERSION = "1.1.0"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checker(size):
    yy, xx = np.indices((size[1], size[0]))
    a = np.where(((xx // 16 + yy // 16) % 2)[..., None],
                 np.array([52, 57, 65, 255]), np.array([78, 83, 91, 255]))
    return Image.fromarray(a.astype(np.uint8), "RGBA")


class Rig:
    def __init__(self, config, part_ids, size, only):
        self.config = config
        self.size = size
        self.nodes = config["nodes"]
        self.mapping = config["part_nodes"]
        self.controls = {}
        missing = sorted(set(part_ids) - self.mapping.keys())
        if missing:
            raise ValueError("Every part must have an explicit node (null for fixed): " + ", ".join(missing))
        self.active = set(self.nodes)
        if only:
            self.active = set()
            for pattern in only.split(","):
                matches = {p for p in part_ids if fnmatch.fnmatch(p, pattern)}
                matches |= {n for n in self.nodes if fnmatch.fnmatch(n, pattern)}
                if not matches:
                    raise ValueError("No part or control matches --only " + pattern)
                for p in matches:
                    node = p if p in self.nodes else self.mapping[p]
                    if node is not None:
                        self.active.add(node)
        for node in self.nodes:
            self.chain(node)  # validate cycles and references before any render
        for pid in part_ids:
            self.chain(self.mapping[pid])

    def set_pose(self, controls):
        """Explicit per-control values override the global diagnostic drive."""
        if set(controls) - self.nodes.keys():
            raise ValueError("Unknown pose control: " + str(set(controls) - self.nodes.keys()))
        for name, values in controls.items():
            if set(values) - {"degrees", "drive", "translation_px"}:
                raise ValueError("Unknown pose parameter for " + name)
            if "translation_px" in values:
                if self.nodes[name]["type"] != "translate":
                    raise ValueError("translation_px requires a translate node")
                vector = np.asarray(values["translation_px"], float)
                if vector.shape != (2,) or not np.isfinite(vector).all():
                    raise ValueError("translation_px must be a finite source-pixel pair")
            for key in ("degrees", "drive"):
                if key in values and not math.isfinite(float(values[key])):
                    raise ValueError("Non-finite pose parameter")
        self.controls = controls

    def chain(self, node):
        result = []
        while node:
            if node in result:
                raise ValueError("Rig parent cycle: " + str(result))
            if node not in self.nodes:
                raise ValueError("Unknown rig node " + node)
            result.append(node)
            node = self.nodes[node].get("parent")
        return result  # deepest first: local deformation, then ancestors

    def local(self, name, points, degrees, drive):
        n = self.nodes[name]
        if name not in self.active:
            return points.copy()
        values = self.controls.get(name, {})
        degrees, drive = values.get("degrees", degrees), values.get("drive", drive)
        if n["type"] == "translate":
            delta = values.get("translation_px", np.asarray(n.get("amplitude_px", [0, 0])) * drive)
            return points + np.asarray(delta, dtype=float)
        root = np.asarray(n["root"], dtype=float)
        p = points - root
        kind = n["type"]
        if kind == "scale":
            return root + p * (1 + drive * np.asarray(n["amplitude_fraction"]))
        theta = math.radians(degrees * n.get("gain", 1))
        if kind == "bend":
            axis = np.asarray(n["tip"], dtype=float) - root
            t = np.clip((p @ axis) / (axis @ axis), 0, 1)
            lock = n.get("lock_fraction", .12)
            t = np.clip((t - lock) / (1 - lock), 0, 1)
            theta = theta * t ** n.get("power", 1.5)
        elif kind != "rotate":
            raise ValueError("Unknown node type " + kind)
        c, s = np.cos(theta), np.sin(theta)
        return root + np.stack((p[..., 0] * c - p[..., 1] * s,
                                p[..., 0] * s + p[..., 1] * c), axis=-1)

    def forward(self, part, points, degrees, drive):
        out = np.array(points, dtype=float)
        for name in self.chain(self.mapping[part]):
            out = self.local(name, out, degrees, drive)
        return out

    def inverse(self, part, points, degrees, drive):
        # Fixed-point inversion solves y = x + displacement(x); degrees <=4.
        # Unlike subtracting displacement(y) once, this records residual error.
        out = np.array(points, dtype=float)
        for _ in range(16):
            error = points - self.forward(part, out, degrees, drive)
            out += error
            if np.max(np.abs(error), initial=0) < 1e-6:
                break
        residual = np.max(np.linalg.norm(self.forward(part, out, degrees, drive) - points, axis=-1), initial=0)
        return out, float(residual)

    def parameters(self, degrees, drive):
        return {name: {
            **n,
            "active": name in self.active,
            "applied_local_degrees": self.controls.get(name, {}).get("degrees", degrees) * n.get("gain", 1) if name in self.active and n["type"] in ("rotate", "bend") else 0,
            "applied_scale": (1 + self.controls.get(name, {}).get("drive", drive) * np.asarray(n["amplitude_fraction"])).tolist()
                if name in self.active and n["type"] == "scale" else [1, 1],
            "applied_translation_px": np.asarray(self.controls.get(name, {}).get("translation_px",
                np.asarray(n.get("amplitude_px", [0, 0])) * self.controls.get(name, {}).get("drive", drive))).tolist()
                if name in self.active and n["type"] == "translate" else [0, 0],
        } for name, n in self.nodes.items()}


def warp(rig, pid, im, degrees, drive, step):
    bbox = im.getchannel("A").getbbox()
    if not bbox:
        return None, {"empty": True}
    if (degrees == 0 and drive == 0 and not rig.controls) or not any(n in rig.active for n in rig.chain(rig.mapping[pid])):
        return (im.crop(bbox), bbox[:2]), {"sampled_max_displacement_px": 0, "inverse_residual_px": 0, "mesh_center_residual_px": 0, "conservative_bbox_leaves_canvas": False}
    x0, y0, x1, y1 = bbox
    xx, yy = np.meshgrid(np.arange(x0, x1 + step, step), np.arange(y0, y1 + step, step))
    sample = np.stack((np.minimum(xx, x1), np.minimum(yy, y1)), axis=-1)
    moved = rig.forward(pid, sample, degrees, drive)
    lo = np.floor(moved.reshape(-1, 2).min(axis=0) - step).astype(int)
    hi = np.ceil(moved.reshape(-1, 2).max(axis=0) + step).astype(int)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, rig.size)
    width, height = (hi - lo).tolist()
    if width <= 0 or height <= 0:
        return None, {"entirely_outside_canvas": True, "conservative_bbox_leaves_canvas": True}
    mesh, corners, boxes = [], [], []
    for y in range(0, height, step):
        for x in range(0, width, step):
            xr, yb = min(width, x + step), min(height, y + step)
            boxes.append((x, y, xr, yb))
            corners.extend(((x + lo[0], y + lo[1]), (x + lo[0], yb + lo[1]),
                            (xr + lo[0], yb + lo[1]), (xr + lo[0], y + lo[1])))
    inv, residual = rig.inverse(pid, np.asarray(corners), degrees, drive)
    source_centers = inv.reshape(-1, 4, 2).mean(axis=1)
    target_centers = np.asarray(corners).reshape(-1, 4, 2).mean(axis=1)
    mesh_error = np.linalg.norm(rig.forward(pid, source_centers, degrees, drive) - target_centers, axis=-1)
    if mesh_error.max(initial=0) > .25 and step > 4:
        result, stats = warp(rig, pid, im, degrees, drive, max(4, step // 2))
        stats["requested_mesh_step_px"] = step
        return result, stats
    for box, quad in zip(boxes, inv.reshape(-1, 8)):
        mesh.append((box, tuple(quad)))
    # Premultiplied RGBA avoids dark/coloured fringes from invisible RGB values.
    transformed = im.convert("RGBa").transform((width, height), Image.Transform.MESH,
                   mesh, Image.Resampling.BICUBIC).convert("RGBA")
    # The transformed alpha bounding rectangle is conservative: transparent
    # corners may leave the canvas without actual artwork being clipped.
    true_clip = bool(np.any(moved.reshape(-1, 2).min(axis=0) < 0) or
                     np.any(moved.reshape(-1, 2).max(axis=0) > rig.size))
    return (transformed, tuple(map(int, lo))), {
        "sampled_max_displacement_px": round(float(np.max(np.linalg.norm(moved - sample, axis=-1))), 4),
        "inverse_residual_px": residual,
        "mesh_center_residual_px": float(mesh_error.max(initial=0)),
        "actual_mesh_step_px": step,
        "conservative_bbox_leaves_canvas": true_clip,
        "local_chain_deepest_first": rig.chain(rig.mapping[pid]),
    }


def composite(rig, parts, order, degrees, drive, step):
    out = Image.new("RGBA", rig.size)
    stats = {}
    for pid in order:
        result, stats[pid] = warp(rig, pid, parts[pid], degrees, drive, step)
        if result:
            out.alpha_composite(*result)
    return out, stats


def display(im, background):
    out = background.copy()
    out.alpha_composite(im)
    return out.convert("RGB")


def sheet(images, labels, target, columns=5, tile=(282, 540)):
    rows = math.ceil(len(images) / columns)
    out = Image.new("RGB", (tile[0] * columns, tile[1] * rows), (35, 39, 46))
    draw = ImageDraw.Draw(out)
    for i, (im, label) in enumerate(zip(images, labels)):
        x, y = (i % columns) * tile[0], (i // columns) * tile[1]
        cut = im.copy()
        cut.thumbnail((tile[0] - 10, tile[1] - 36), Image.Resampling.LANCZOS)
        out.paste(cut, (x + (tile[0] - cut.width) // 2, y + 30))
        draw.text((x + 7, y + 6), label, fill="white", font=font(14))
    out.save(target)


def render_cases(case_path, project_path, manifest, part_root, config, output, background, step):
    """Render every registered relative pose; outputs never assert visual pass."""
    from relations import dependency_fingerprint, state_visibility, validate_relations
    from asset_pipeline import register_image

    project = read_json(project_path)
    cases_doc = read_json(case_path)
    size = tuple(config["canvas"])
    source_path = project_path.parent / project["source"]
    with Image.open(source_path) as source_image:
        if source_image.size != size:
            raise ValueError("Rig canvas differs from the source illustration")
    relation_check = validate_relations(project, size)
    items = manifest["parts"] if isinstance(manifest, dict) else manifest
    if any(m.get("status", "ready") != "ready" for m in items):
        raise ValueError("Relative tests require every drawing including alternate states")
    ids = [m["id"] for m in items]
    if len(ids) != len(set(ids)) or set(ids) != {p["id"] for p in project["parts"]}:
        raise ValueError("Relative-test manifest must match all project drawing IDs")
    files = {m["id"]: part_root / m.get("file", m["id"] + ".png") for m in items}
    parts = {pid: Image.open(path).convert("RGBA") for pid, path in files.items()}
    if any(im.size != size for im in parts.values()):
        raise ValueError("Relative-test parts must share the source canvas")
    source_asset_hashes = {}
    for part in project["parts"]:
        source_asset = project_path.parent / part["asset"]
        with Image.open(source_asset) as raw:
            expected, _, _, _, _ = register_image(raw, part, size)
        if not np.array_equal(np.asarray(expected), np.asarray(parts[part["id"]])):
            raise ValueError("Registered drawing is stale relative to project asset: " + part["id"])
        source_asset_hashes[part["id"]] = sha256(source_asset)
    order = config.get("order", [m["id"] for m in sorted(items, key=lambda m: m.get("order", 0))])
    if set(order) != set(ids) or len(order) != len(ids):
        raise ValueError("Relative-test rig order must include all drawings, including hidden states")
    # A different rig order could undo the ordering checked by compose.
    project_order = [p["id"] for p in sorted(project["parts"], key=lambda p: p.get("order", project["parts"].index(p)))]
    if order != project_order:
        raise ValueError("Relative-test rig order must match the validated project order")
    rig = Rig(config, ids, size, None)
    hashes = {pid: sha256(path) for pid, path in files.items()}
    hashes["__source__"] = sha256(source_path)
    cases = cases_doc.get("cases", [])
    case_ids = [c["id"] for c in cases]
    if not cases or len(case_ids) != len(set(case_ids)):
        raise ValueError("cases must be nonempty with unique IDs")
    declared = {c["id"] for c in project.get("test_cases", [])}
    unknown = set(case_ids) - declared
    if unknown:
        raise ValueError("Test cases must be registered in project.test_cases: " + str(sorted(unknown)))
    by_relation = {r["id"]: r for r in project.get("relations", [])}
    results = []
    output.mkdir(parents=True, exist_ok=True)
    for case in cases:
        cid = case["id"]
        if not re.fullmatch(r"[\w-]+", cid):
            raise ValueError("Use filename-safe case IDs")
        relations = case.get("relation_ids", [])
        if not relations or set(relations) - by_relation.keys():
            raise ValueError("Every case must reference existing relations")
        if any(cid not in by_relation[r].get("test_cases", []) for r in relations):
            raise ValueError("Relation does not declare this case")
        state = case.get("state_id", "default")
        poses = case.get("poses", [])
        pose_ids = [p["id"] for p in poses]
        if not poses or len(pose_ids) != len(set(pose_ids)):
            raise ValueError("Every case needs unique explicit poses; no endpoint is inferred")
        roi = case.get("source_roi", [0, 0, *size])
        if (len(roi) != 4 or any(int(x) != x for x in roi)
                or not (0 <= roi[0] < roi[2] <= size[0] and 0 <= roi[1] < roi[3] <= size[1])):
            raise ValueError("source_roi must be an in-canvas integer source rectangle")
        pose_states = sorted({state, *(pose.get("state_id", state) for pose in poses)})
        relevant = {by_relation[r][key] for r in relations for key in ("front", "back")}
        # A context layer can reveal a defect; include all displayed dependencies.
        for pose_state in pose_states:
            relevant |= state_visibility(project, pose_state)
        fingerprint = dependency_fingerprint(project, sorted(relevant), hashes, rig=config,
            relation_ids=relations, state_ids=pose_states, evidence_kind="motion")
        frames, views, labels = [], [], []
        for pose in poses:
            if not re.fullmatch(r"[\w-]+", pose["id"]):
                raise ValueError("Use filename-safe pose IDs")
            pose_state = pose.get("state_id", state)
            visible = state_visibility(project, pose_state)
            hidden = set(pose.get("hide", []))
            if hidden - set(ids):
                raise ValueError("Unknown hidden drawing")
            rig.set_pose(pose.get("controls", {}))
            degrees, drive = float(pose.get("degrees", 0)), float(pose.get("drive", 0))
            if not np.isfinite([degrees, drive]).all():
                raise ValueError("Non-finite global pose drive")
            selected = [pid for pid in order if pid in visible and pid not in hidden]
            im, stats = composite(rig, parts, selected, degrees, drive, step)
            stem = cid + "--" + pose["id"]
            full = output / (stem + ".png")
            im.save(full)
            view = display(im, background).crop(roi)
            crop = output / (stem + "--roi.png")
            view.save(crop)
            views.append(view)
            labels.append(pose["id"])
            frames.append({"id": pose["id"], "state_id": pose_state, "hidden": sorted(hidden),
                "render_order": selected, "parameters": rig.parameters(degrees, drive),
                "foreground": {"path": output.name + "/" + full.name, "sha256": sha256(full)},
                "roi": {"path": output.name + "/" + crop.name, "sha256": sha256(crop)},
                "parts": stats, "visual_status": "pending"})
        sheet(views, labels, output / (cid + "--contact.png"), columns=min(5, len(views)), tile=(320, 360))
        results.append({"id": cid, "relation_ids": relations, "source_roi": roi,
            "required_frame_ids": pose_ids, "rendered_frames": len(frames), "frames": frames,
            "dependency_fingerprint": fingerprint, "visual_status": "pending"})
    return {"status": "diagnostic_only_requires_visual_review", "cases": results,
        "missing_registered_case_ids": sorted(declared - set(case_ids)),
        "relation_validation": relation_check, "cases_sha256": sha256(case_path),
        "project_sha256": sha256(project_path), "asset_sha256": hashes,
        "source_asset_sha256": source_asset_hashes,
        "note": "Every explicit pose is saved. Render counts are not reviewed test coverage. No iris aperture or continuous eyelid anatomy is synthesized."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parts", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--rig", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--background", type=Path)
    p.add_argument("--reference", type=Path, help="Optional Photoshop static export for visible RGB comparison")
    p.add_argument("--only", help="Comma-separated part/control IDs or globs; children inherit selected controls")
    p.add_argument("--frames", type=int, default=16, help="Loop frame count; 0 disables GIF")
    p.add_argument("--mesh-step", type=int, default=20, help="Inverse mesh spacing in full-resolution pixels")
    p.add_argument("--probes", help="Extra single-control +/-4 degree contacts, comma-separated controls")
    p.add_argument("--cases", type=Path, help="Explicit registered relative poses, including source-pixel translation and state previews")
    p.add_argument("--project", type=Path, help="Production project with relations and states; required with --cases")
    p.add_argument("--cases-only", action="store_true", help="Render explicit cases without the legacy global diagnostic")
    args = p.parse_args()
    if args.cases and not args.project or args.cases_only and not args.cases:
        p.error("--cases requires --project; --cases-only requires --cases")
    if args.frames not in (0,) and args.frames < 8:
        p.error("Use --frames 0 or at least 8")
    if args.mesh_step < 4:
        p.error("--mesh-step must be >=4")
    items = read_json(args.manifest)
    full_manifest = items
    if isinstance(items, dict):
        items = items["parts"]
    incomplete = [m["id"] for m in items if m.get("status", "ready") != "ready"]
    if incomplete:
        raise ValueError("Incomplete asset manifest; register all expected parts first: " + ", ".join(incomplete))
    hidden = {m["id"] for m in items if not m.get("visible", True)}
    items = [m for m in items if m.get("visible", True)]
    if not items:
        raise ValueError("No visible parts in the selected default pose")
    ids = [m["id"] for m in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate part IDs")
    files = {m["id"]: args.parts / m.get("file", m["id"] + ".png") for m in items}
    parts = {pid: Image.open(path).convert("RGBA") for pid, path in files.items()}
    size = next(iter(parts.values())).size
    if any(im.size != size for im in parts.values()):
        raise ValueError("All parts must be aligned full-canvas RGBA files")
    config = read_json(args.rig)
    if list(size) != config["canvas"]:
        raise ValueError("Rig canvas differs from input; adapt actual coordinates explicitly")
    order = config.get("order", [m["id"] for m in sorted(items, key=lambda m: m.get("order", 0))])
    order = [pid for pid in order if pid not in hidden]
    if set(order) != set(ids) or len(order) != len(ids):
        raise ValueError("Rig order must include each input part exactly once")
    rig = Rig(config, ids, size, args.only)
    args.out.mkdir(parents=True, exist_ok=True)
    bg = Image.open(args.background).convert("RGBA") if args.background else checker(size)
    if bg.size != size:
        raise ValueError("Background differs from canvas")
    relative_report = None
    if args.cases:
        relative_report = render_cases(args.cases, args.project, full_manifest, args.parts,
            config, args.out / "relative", bg, args.mesh_step)
        relative_report["rig_sha256"] = sha256(args.rig)
        (args.out / "relative-motion-report.json").write_text(json.dumps(relative_report,
            ensure_ascii=False, indent=2), encoding="utf-8")
        if args.cases_only:
            print(str((args.out / "relative-motion-report.json").resolve()), flush=True)
            return
    static, static_stats = composite(rig, parts, order, 0, 0, args.mesh_step)
    static.save(args.out / "static-foreground.png")
    static_display = display(static, bg)
    static_display.save(args.out / "static.png")
    opaque = static.getchannel("A").point(lambda a: 255 if a >= 250 else 0)
    interior = np.asarray(opaque.filter(ImageFilter.MinFilter(19))) > 0
    report = {
        "tool": "motion_check", "version": VERSION,
        "status": "diagnostic_only_requires_visual_review",
        "method": "Each original RGBA layer has its own rig chain; nonlinear root-fixed bend + rigid elbow rotation + inherited torso/head transforms; numerical inverse mesh",
        "canvas": size, "mesh_step_px": args.mesh_step,
        "excluded_default_hidden_parts": sorted(hidden),
        "mesh_refinement": "Starts at requested step; halves spacing until sampled cell-center forward error <=0.25px or spacing=4px. Per-layer report stores actual spacing and residual. This is a sampled numerical check, not a global error bound.",
        "positive_rotation": "clockwise in image coordinates",
        "root_lock_note": "Bend root and proximal lock_fraction of its axis are fixed in the parent frame. Endpoint control drive is multiplied by each node gain to give LOCAL degrees; children may have larger world displacement due to inherited motion. Consult each frame parameters, not the drive label, for the actual per-node angle.",
        "gap_metric_note": "A candidate is alpha <26 inside the original opaque union eroded by9px. This is an exposure diagnostic, not an automatic failure/pass: intended moving silhouettes can also expose pixels.",
        "source": {"manifest": str(args.manifest.resolve()), "manifest_sha256": sha256(args.manifest),
                   "rig": str(args.rig.resolve()), "rig_sha256": sha256(args.rig),
                   "parts": {pid: {"path": str(files[pid].resolve()), "sha256": sha256(files[pid]),
                                   "bbox": parts[pid].getchannel("A").getbbox()} for pid in ids}},
        "only": args.only, "render_order_back_to_front": order,
        "frames": [], "loop_frames": [], "probes": [],
        "relative_tests": relative_report,
    }
    if args.reference:
        ref = np.asarray(Image.open(args.reference).convert("RGB"), dtype=int)
        if ref.shape != np.asarray(static_display).shape:
            raise ValueError("Reference canvas mismatch")
        diff = np.abs(ref - np.asarray(static_display, dtype=int))
        report["static_reference"] = {"path": str(args.reference.resolve()),
            "mean_absolute_rgb_error": float(diff.mean()),
            "pixels_any_channel_difference_over_8": int(np.count_nonzero(diff.max(axis=2) > 8)),
            "note": "Nonzero differences can indicate stale part exports, order/blend-mode differences, or a different background; investigate before certifying motion."}
    images, labels, highlights = [], [], []
    for label, degrees, drive in [("static", 0, 0), ("minus2", -2, -.5), ("plus2", 2, .5),
                                   ("minus4", -4, -1), ("plus4", 4, 1)]:
        if degrees == 0:
            im, stats = static, static_stats
        else:
            im, stats = composite(rig, parts, order, degrees, drive, args.mesh_step)
        im.save(args.out / (label + "-foreground.png"))
        view = display(im, bg)
        view.save(args.out / (label + ".png"))
        candidates = interior & (np.asarray(im.getchannel("A")) < 26)
        heat = np.asarray(view).copy()
        heat[candidates] = [255, 0, 180]
        highlight = Image.fromarray(heat)
        highlight.save(args.out / (label + "-exposure.png"))
        images.append(view)
        torso_percent = drive * max(config["nodes"].get("breath", {}).get("amplitude_fraction", [0, 0])) * 100 if "breath" in rig.active else 0
        labels.append(f"drive {degrees:+g} deg / torso {torso_percent:+g}%")
        highlights.append(highlight)
        report["frames"].append({"file": label + ".png", "drive": drive,
            "degrees": degrees, "parameters": rig.parameters(degrees, drive),
            "new_interior_exposure_pixels": int(candidates.sum()), "parts": stats})
        print(f"{label}: exposure candidates={candidates.sum()}", flush=True)
    sheet(images, labels, args.out / "endpoints-contact.png")
    sheet(highlights, labels, args.out / "exposure-contact.png")
    cuts = []
    for pid in order:
        box = parts[pid].getchannel("A").getbbox()
        cut = parts[pid].crop(box) if box else Image.new("RGBA", (1, 1))
        cuts.append(display(cut, checker(cut.size)))
    sheet(cuts, order, args.out / "isolated-parts-contact.png", columns=6, tile=(230, 260))
    # Full-resolution crops retain details that the full-figure contact can hide.
    region_images, region_labels = [], []
    for name, box in config.get("review_regions", {}).items():
        for index in (0, 3, 4):
            region_images.append(images[index].crop(box))
            region_labels.append(name + " " + labels[index])
    if region_images:
        sheet(region_images, region_labels, args.out / "regions-contact.png", columns=3, tile=(380, 430))
    if args.frames:
        loop = []
        for i in range(args.frames):
            phase = i * math.tau / args.frames
            drive = math.sin(phase)
            degrees = 4 * drive
            im, stats = composite(rig, parts, order, degrees, drive, args.mesh_step)
            frame = display(im, bg)
            frame.thumbnail((565, 1004), Image.Resampling.LANCZOS)
            torso_percent = drive * max(config["nodes"].get("breath", {}).get("amplitude_fraction", [0, 0])) * 100 if "breath" in rig.active else 0
            ImageDraw.Draw(frame).text((8, 8), f"drive {degrees:+.2f} deg / torso {torso_percent:+.2f}%", font=font(16), fill=(255, 0, 100), stroke_width=1, stroke_fill="white")
            loop.append(frame)
            report["loop_frames"].append({"index": i, "phase_radians": phase,
                "degrees": degrees, "drive": drive, "parameters": rig.parameters(degrees, drive),
                "parts": stats})
        loop[0].save(args.out / "motion-loop.gif", save_all=True, append_images=loop[1:],
                     duration=100, loop=0, disposal=2)
    if args.probes:
        probe_images, probe_labels = [], []
        for control in args.probes.split(","):
            single = Rig(config, ids, size, control)
            box = config.get("probe_regions", {}).get(control)
            if not box:
                affected = [parts[pid].getchannel("A").getbbox() for pid in ids
                    if any(n in single.active for n in single.chain(single.mapping[pid]))]
                affected = [b for b in affected if b]
                if not affected:
                    raise ValueError("Selected control affects no visible pixels: " + control)
                box = [max(0, min(b[0] for b in affected)-30), max(0, min(b[1] for b in affected)-30),
                       min(size[0], max(b[2] for b in affected)+30), min(size[1], max(b[3] for b in affected)+30)]
            records = []
            for degrees in (-4, 0, 4):
                im, stats = composite(single, parts, order, degrees, degrees / 4, args.mesh_step)
                view = display(im, bg)
                filename = f"probe-{control}-{degrees:+g}.png"
                view.save(args.out / filename)
                probe_images.append(view.crop(box))
                probe_labels.append(control + f" {degrees:+g} deg")
                records.append({"degrees": degrees, "file": filename, "parts": stats,
                                "parameters": single.parameters(degrees, degrees/4)})
            report["probes"].append({"control": control, "crop": box, "frames": records})
        sheet(probe_images, probe_labels, args.out / "single-controls-contact.png", columns=3, tile=(380, 430))
    (args.out / "motion-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(str((args.out / "motion-report.json").resolve()), flush=True)


if __name__ == "__main__":
    main()
