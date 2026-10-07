# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 ChitoseRen
"""Inspect source-pixel ownership before generating a reconstruction candidate.

Usage: python mask_preflight.py reconstruction.json
Pillow and numpy only. Does not infer semantics, draw hidden content, or approve art.
inspect_config(path) performs read-only inspection for reconstruct_parts.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(root, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("Paths must be nonempty and relative to the config directory")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Path leaves the config directory: " + value)
    return path


def load_image(path, size=None, mode=None):
    with Image.open(path) as raw:
        if raw.info.get("icc_profile"):
            raise ValueError("Convert an inspected copy to sRGB and remove ICC metadata first: " + path.name)
        if mode and raw.mode != mode:
            raise ValueError(f"{path.name} must be {mode}, got {raw.mode}")
        if not mode and raw.mode not in ("RGB", "RGBA"):
            raise ValueError(path.name + " must be RGB or RGBA")
        if size and raw.size != tuple(size):
            raise ValueError(path.name + " must have the source canvas dimensions")
        return raw.copy()


def load_binary(path, size):
    array = np.asarray(load_image(path, size, "L"))
    if np.any((array != 0) & (array != 255)):
        raise ValueError(path.name + " must contain only 0 and 255")
    return array == 255


def reject_constant(value):
    raise ValueError("Non-finite JSON number: " + value)


def _inspect(filename):
    filename = Path(filename).resolve()
    root = filename.parent
    errors, hashes, loaded, records = [], {}, [], []
    context = {"root": root, "filename": filename, "loaded": loaded}
    report = {"status": "invalid", "errors": errors, "parts": records,
              "visual_review": "required; no semantic pass is generated"}
    selected = {}
    try:
        config = json.loads(filename.read_text(encoding="utf-8-sig"), parse_constant=reject_constant)
        if not isinstance(config, dict):
            raise ValueError("Config must be an object")
        context["config"] = config
        source_path = resolve(root, config["source"])
        hashes[config["source"]] = digest(source_path)
        source = load_image(source_path).convert("RGBA")
        context["source"] = source
        size = source.size
        source_array = np.asarray(source)
        report["canvas"] = list(size)
        # Only pre-generation inputs bind this review. Candidates and final
        # reconstruction masks intentionally arrive after source inspection.
        selected = {"source": config["source"], "parts": [],
                    "critical_interface_ids": config.get("critical_interface_ids"),
                    "interfaces": config.get("interfaces")}
        parts = config.get("parts")
        if not isinstance(parts, list) or not parts:
            raise ValueError("parts must be a nonempty list")
        ids = [part.get("id") if isinstance(part, dict) else None for part in parts]
        if any(not isinstance(pid, str) or not re.fullmatch(r"[\w-]+", pid) for pid in ids):
            raise ValueError("Each part needs a stable id of letters, digits, underscores or hyphens")
        reserved = {"con", "prn", "aux", "nul", *("com" + str(i) for i in range(10)), *("lpt" + str(i) for i in range(10))}
        if len({pid.casefold() for pid in ids}) != len(ids) or any(pid.casefold() in reserved for pid in ids):
            raise ValueError("Part ids must be case-insensitively unique and not Windows device names")
        ownership_count = np.zeros((size[1], size[0]), np.uint32)
        context["overlap"] = ownership_count > 1
        for index, part in enumerate(parts):
            pid = part["id"]
            fields = {key: part.get(key) for key in ("id", "source_layer", "keep", "unknown", "edit")}
            fields["order"] = part.get("order", index)
            selected["parts"].append(fields)
            item = {"id": pid, "errors": [], "preview": None}
            records.append(item)
            try:
                if isinstance(fields["order"], bool) or not isinstance(fields["order"], (int, float)) or not np.isfinite(fields["order"]):
                    raise ValueError("order must be a finite number")
                paths = {key: resolve(root, fields[key]) for key in ("source_layer", "keep", "unknown", "edit")}
                for key, path in paths.items():
                    hashes[fields[key]] = digest(path)
                layer = load_image(paths["source_layer"], size, "RGBA")
                masks = {key: load_binary(paths[key], size) for key in ("keep", "unknown", "edit")}
                a = np.asarray(layer)
                keep, unknown, edit = (masks[key] for key in ("keep", "unknown", "edit"))
                conditions = {
                    "keep_overlaps_edit": keep & edit,
                    "keep_overlaps_unknown": keep & unknown,
                    "keep_not_opaque": keep & ((a[..., 3] != 255) | (source_array[..., 3] != 255)),
                    "keep_rgb_changed": keep & np.any(a[..., :3] != source_array[..., :3], axis=2),
                    "mixed_alpha_outside_unknown": (a[..., 3] > 0) & (a[..., 3] < 255) & ~unknown,
                    "unclassified_visible_pixels": (a[..., 3] > 0) & ~(keep | unknown | edit),
                }
                for label, values in conditions.items():
                    count = int(np.count_nonzero(values))
                    item[label] = count
                    if count:
                        item["errors"].append(f"{label}: {count} pixels")
                ownership_count += keep.astype(np.uint32)
                loaded.append({"id": pid, "order": fields["order"], "index": index,
                               "image": layer, **masks})
                item["keep_pixels"] = int(keep.sum())
                item["unknown_pixels"] = int(unknown.sum())
                item["edit_pixels"] = int(edit.sum())
            except (OSError, ValueError, KeyError, TypeError) as exc:
                item["errors"].append(str(exc))
            errors.extend(pid + ": " + e for e in item["errors"])
        overlap = ownership_count > 1
        context["overlap"] = overlap
        report["keep_ownership_overlap_pixels"] = int(overlap.sum())
        if overlap.any():
            errors.append("Trusted source ownership overlaps between parts; inspect both layers")

        interfaces = config.get("interfaces", [])
        critical = config.get("critical_interface_ids", [])
        if not isinstance(interfaces, list) or not isinstance(critical, list):
            raise ValueError("interfaces and critical_interface_ids must be lists")
        if not critical or any(not isinstance(i, str) or not i for i in critical) or len(set(critical)) != len(critical):
            raise ValueError("critical_interface_ids must be a nonempty unique string list from visual inventory")
        interface_ids = [i.get("id") if isinstance(i, dict) else None for i in interfaces]
        if any(not isinstance(i, str) or not i for i in interface_ids) or len(set(interface_ids)) != len(interface_ids):
            raise ValueError("Every interface needs a unique id")
        missing = set(critical) - set(interface_ids)
        if missing:
            errors.append("Missing critical interfaces: " + ", ".join(sorted(missing)))
        covered_parts = set()
        for interface in interfaces:
            iid = interface["id"]
            involved = interface.get("parts")
            if not isinstance(involved, list) or not involved or any(not isinstance(p, str) or p not in ids for p in involved) or len(set(involved)) != len(involved):
                errors.append(iid + ": parts must reference distinct local part ids")
                continue
            roi = interface.get("roi")
            if not isinstance(roi, list) or len(roi) != 4 or any(type(v) is not int for v in roi) or not (0 <= roi[0] < roi[2] <= size[0] and 0 <= roi[1] < roi[3] <= size[1]):
                errors.append(iid + ": roi must be an integer source-canvas xyxy rectangle")
            kind = interface.get("kind")
            if kind not in ("outer", "occlusion", "internal", "material", "shadow"):
                errors.append(iid + ": unsupported boundary kind")
            if kind in ("outer", "occlusion") and interface.get("front") not in involved:
                errors.append(iid + ": an observed front part is required")
            if kind == "occlusion" and len(involved) != 2:
                errors.append(iid + ": an occlusion needs two parts")
            if interface.get("line_owner") not in involved + ["none"]:
                errors.append(iid + ": line_owner must name an involved part or none")
            for field in ("observation", "hidden_requirement"):
                if not isinstance(interface.get(field), str) or not interface[field].strip():
                    errors.append(iid + ": " + field + " needs an actual observation")
            if iid in critical:
                covered_parts.update(involved)
                if interface.get("observed") is not True:
                    errors.append(iid + ": critical relationship is not yet observed")
            if kind == "occlusion" and interface.get("front") in involved:
                effective = {p["id"]: (p["order"], p["index"]) for p in loaded}
                front = interface["front"]
                back = next(p for p in involved if p != front)
                if front in effective and back in effective and effective[front] <= effective[back]:
                    errors.append(iid + ": front/back contradict the local drawing order")
        for pid in set(ids) - covered_parts:
            errors.append(pid + ": no inventoried critical interface; include its true outer boundary when isolated")
        report["critical_interfaces"] = {"expected": len(critical), "missing": sorted(missing)}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(str(exc))
    fingerprint_data = {"definition": selected, "inputs": hashes}
    report["fingerprint"] = hashlib.sha256(json.dumps(fingerprint_data, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()
    report["input_sha256"] = hashes
    report["status"] = "invalid" if errors else "requires_visual_review"
    return report, context


def inspect_config(filename):
    """Return a JSON-safe read-only report with fingerprint and errors list."""
    return _inspect(filename)[0]


def write_previews(filename):
    report, context = _inspect(filename)
    if "source" not in context or "config" not in context:
        return report
    root, config = context["root"], context["config"]
    out = resolve(root, config.get("output", "reconstruction")) / "preflight"
    if out == root or root.is_relative_to(out):
        raise ValueError("Output must be a project subdirectory")
    paths = [out / "preflight-report.json", out / "source-only.png", out / "comparison.png", out / "ownership-conflicts.png"]
    paths.extend(out / "parts" / (p["id"] + ".png") for p in context["loaded"])
    paths.extend(out / ("roi-" + str(i) + ".png") for i, _ in enumerate(config.get("interfaces", [])))
    protected = {context["filename"], *(resolve(root, p) for p in report["input_sha256"])}
    # Also protect declared artistic inputs that are not used until generation.
    for part in config.get("parts", []):
        for field in ("candidate", "target_alpha", "generated_weight", "blend_allowed", "color_protect"):
            if part.get(field):
                protected.add(resolve(root, part[field]))
    for group in config.get("appearance_groups", []):
        protected.add(resolve(root, group["region"]))
        protected.update(resolve(root, sample["mask"]) for sample in group.get("samples", []))
    if any(p.resolve() in protected for p in paths):
        raise ValueError("Preflight output would overwrite an input")
    if any(not p.resolve().is_relative_to(root) for p in paths):
        raise ValueError("Resolved output leaves the project directory")
    out.mkdir(parents=True, exist_ok=True)
    (out / "parts").mkdir(exist_ok=True)
    source = context["source"]
    assembled = Image.new("RGBA", source.size)
    for part in sorted(context["loaded"], key=lambda p: (p["order"], p["index"])):
        assembled.alpha_composite(part["image"])
        preview = Image.new("RGB", (source.width * 3, source.height), "white")
        for index, color in enumerate(("black", "white")):
            backdrop = Image.new("RGBA", source.size, color)
            backdrop.alpha_composite(part["image"])
            preview.paste(backdrop.convert("RGB"), (index * source.width, 0))
        labels = np.zeros((source.height, source.width, 3), np.uint8)
        labels[part["edit"]] = (215, 75, 70)
        labels[part["unknown"]] = (235, 185, 35)
        labels[part["keep"]] = (40, 180, 100)
        preview.paste(Image.fromarray(labels), (2 * source.width, 0))
        preview.save(out / "parts" / (part["id"] + ".png"))
        next(p for p in report["parts"] if p["id"] == part["id"])["preview"] = "parts/" + part["id"] + ".png"
    assembled.save(out / "source-only.png")
    gray = Image.new("RGBA", source.size, (128, 128, 128, 255))
    gray.alpha_composite(assembled)
    src = Image.new("RGBA", source.size, (128, 128, 128, 255))
    src.alpha_composite(source)
    diff = np.abs(np.asarray(src)[..., :3].astype(int) - np.asarray(gray)[..., :3].astype(int))
    sheet = Image.new("RGB", (source.width * 3, source.height + 22), "white")
    for index, (label, im) in enumerate((("SOURCE", src.convert("RGB")), ("PARTIAL SOURCE LAYERS", gray.convert("RGB")), ("ABS RGB DIFFERENCE (not a pass)", Image.fromarray(diff.astype(np.uint8))))):
        sheet.paste(im, (source.width * index, 22))
        ImageDraw.Draw(sheet).text((source.width * index + 4, 4), label, fill="black")
    sheet.save(out / "comparison.png")
    overlay = np.asarray(src.convert("RGB")).copy()
    overlay[context["overlap"]] = (255, 0, 90)
    Image.fromarray(overlay).save(out / "ownership-conflicts.png")
    for index, interface in enumerate(config.get("interfaces", [])):
        roi = interface.get("roi")
        if isinstance(roi, list) and len(roi) == 4 and all(type(v) is int for v in roi) and 0 <= roi[0] < roi[2] <= source.width and 0 <= roi[1] < roi[3] <= source.height:
            w, h = roi[2] - roi[0], roi[3] - roi[1]
            contact = Image.new("RGB", (w * 2, h))
            contact.paste(src.crop(roi).convert("RGB"), (0, 0))
            contact.paste(gray.crop(roi).convert("RGB"), (w, 0))
            contact.save(out / ("roi-" + str(index) + ".png"))
    report["preview_legend"] = {"green": "trusted original pixels", "yellow": "unknown/edge review", "red": "permitted edit envelope"}
    report["preview_limit"] = "Partial visible-source layers only; inspect isolated layers and source interfaces. No hidden completion or semantic pass."
    (out / "preflight-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Project-relative source reconstruction config")
    args = parser.parse_args()
    try:
        result = write_previews(args.config)
        print(json.dumps({"status": result["status"], "fingerprint": result["fingerprint"],
                          "error_count": len(result["errors"]), "errors": result["errors"][:8]}, ensure_ascii=False))
        return 2 if result["errors"] else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
