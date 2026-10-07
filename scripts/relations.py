# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 ChitoseRen
"""Shared relation validation and evidence dependencies; never awards visual passes."""
from __future__ import annotations

import hashlib
import json
import math

PAINT_ROLES = {"base_color", "linework", "form_shadow", "cast_shadow", "highlight", "texture", "composite"}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _ids(values, label):
    if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError(label + " must be a list of nonempty IDs")
    if len(values) != len(set(values)):
        raise ValueError(label + " contains duplicate IDs")
    return set(values)


def _index(items, label):
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise ValueError(label + " must be a list of objects")
    _ids([item.get("id") for item in items], label + ".id")
    return {item["id"]: item for item in items}


def _finite(values, length, label):
    if (not isinstance(values, list) or len(values) != length
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)):
        raise ValueError(label + " must contain finite coordinates")


def _check_chains(mapping, label):
    for identity in mapping:
        seen, current = set(), identity
        while current is not None:
            if current not in mapping:
                raise ValueError(f"{label}: unknown parent {current}")
            if current in seen:
                raise ValueError(f"{label}: parent cycle through {current}")
            seen.add(current)
            current = mapping[current]


def drawing_parts(project, planning=False):
    """Expand logical planning rows to drawing rows, preserving explicit drawing_specs."""
    if not planning:
        return project["parts"]
    result = []
    for part in project["parts"]:
        for identity in part.get("drawings", [part["id"]]):
            result.append({**part, **project.get("drawing_specs", {}).get(identity, {}),
                           "id": identity, "logical_id": part["id"]})
    return result


def state_visibility(project, state_id="default", planning=False):
    parts = drawing_parts(project, planning)
    default = {p["id"] for p in parts if p.get("visible", True)}
    states = _index(project.get("states", []), "states")
    if state_id in states:
        return _ids(states[state_id].get("visible"), "state.visible")
    if state_id == "default":
        return default
    raise ValueError("Unknown state " + state_id)


def validate_materials(project, drawings):
    """Check declared paint-layer links; whole-image coverage remains a visual judgment."""
    if "materials" not in project:
        if any("material_id" in p or "paint_role" in p for p in drawings.values()):
            raise ValueError("Paint-layer metadata requires a materials inventory")
        return
    materials = _index(project["materials"], "materials")
    if not materials:
        raise ValueError("materials must be nonempty when supplied")
    bases = set()
    for mid, material in materials.items():
        for field in ("name", "base_color_note", "source_evidence"):
            if not isinstance(material.get(field), str) or not material[field].strip():
                raise ValueError(f"{mid}.{field} must describe the observed material")
        ids = _ids(material.get("base_drawings"), mid + ".base_drawings")
        if not ids or ids - drawings.keys():
            raise ValueError(mid + ": base_drawings must name existing drawings")
        for pid in ids:
            part = drawings[pid]
            if part.get("material_id") != mid or part.get("paint_role") != "base_color":
                raise ValueError(pid + ": base drawing must carry the matching material_id and base_color role")
        bases |= ids
    for pid, part in drawings.items():
        role, mid = part.get("paint_role"), part.get("material_id")
        if role is None and mid is None:
            continue  # Legacy/unclassified rows never establish complete visual coverage.
        if role not in PAINT_ROLES or mid not in materials:
            raise ValueError(pid + ": unknown paint_role or material_id")
        if role == "base_color" and pid not in bases:
            raise ValueError(pid + ": base_color is missing from materials.base_drawings")
        receiver = part.get("receiver_id")
        if role in {"form_shadow", "cast_shadow", "highlight", "texture"} or receiver is not None:
            if receiver not in bases or receiver == pid or drawings[receiver]["material_id"] != mid:
                raise ValueError(pid + ": receiver_id must reference a base of the same material")
        if role == "cast_shadow":
            casters = _ids(part.get("caster_ids", []), pid + ".caster_ids")
            if casters - drawings.keys() or pid in casters:
                raise ValueError(pid + ": caster_ids must reference other drawings")
            note = part.get("caster_note")
            if not casters and (not isinstance(note, str) or not note.strip()):
                raise ValueError(pid + ": identify the caster or explain an off-canvas/unknown caster")
            if not isinstance(part.get("motion_policy"), str) or not part["motion_policy"].strip():
                raise ValueError(pid + ": cast_shadow requires an explicit motion_policy")


def validate_relations(project, canvas=None, planning=False):
    """Raise for invalid references/geometry/order; report incomplete inventory as pending.

    No relations is supported for legacy geometry, with coverage unavailable. Logical
    occlusion cycles are allowed: only each state's simultaneous drawing order is checked.
    """
    original = _index(project["parts"], "parts")
    drawings = _index(drawing_parts(project, planning), "drawings")
    validate_materials(project, drawings)
    logical = {p.get("logical_id", p["id"]) for p in drawings.values()}
    warnings = []
    for field in ("semantic_parent", "motion_parent"):
        mapping = {}
        for part in drawings.values():
            identity = part.get("logical_id", part["id"])
            # Legacy parent means semantic containment only. Rig parent stays separate.
            parent = part.get(field, part.get("parent") if field == "semantic_parent" else None)
            if identity in mapping and mapping[identity] != parent:
                raise ValueError(f"{identity}: drawing segments disagree about {field}")
            mapping[identity] = parent
        _check_chains(mapping, field)
    if any("parent" in part for part in original.values()):
        warnings.append("Legacy part.parent is read as semantic_parent only; motion_parent must be explicit.")
    dep_ids = set(original) if planning else set(drawings)
    for part in original.values():
        deps = _ids(part.get("depends_on", []), part["id"] + ".depends_on")
        if deps - dep_ids or part["id"] in deps:
            raise ValueError(part["id"] + ": depends_on must reference other production dependencies")
    # Making an asset cannot depend recursively on itself. This graph is unrelated to occlusion.
    done, active = set(), set()
    def visit(identity):
        if identity in active:
            raise ValueError("depends_on cycle through " + identity)
        if identity not in done:
            active.add(identity)
            for other in original[identity].get("depends_on", []):
                visit(other)
            active.remove(identity)
            done.add(identity)
    for identity in original:
        visit(identity)
    if planning:
        unknown_specs = set(project.get("drawing_specs", {})) - set(drawings)
        if unknown_specs:
            raise ValueError("drawing_specs references unknown drawings: " + str(sorted(unknown_specs)))
    states = _index(project.get("states", []), "states")
    all_states = {"default", *states}
    cases = _index(project.get("test_cases", []), "test_cases")
    for sid, state in states.items():
        visible = _ids(state.get("visible"), sid + ".visible")
        if visible - drawings.keys():
            raise ValueError(sid + ": visible references unknown drawings")
    for cid, case in cases.items():
        if case.get("state_id", "default") not in all_states:
            raise ValueError(cid + ": unknown test case state_id")
    order = {}
    for index, (identity, part) in enumerate(drawings.items()):
        value = part.get("order", index)
        _finite([value], 1, identity + ".order")
        order[identity] = (value, index)
    exclusive = project.get("exclusive_groups", [])
    if not isinstance(exclusive, list):
        raise ValueError("exclusive_groups must be a list of drawing ID lists")
    for group in exclusive:
        if not _ids(group, "exclusive_groups") <= drawings.keys():
            raise ValueError("exclusive_groups references unknown drawings")
    visible_by_state = {sid: state_visibility(project, sid, planning) for sid in sorted(all_states)}
    for sid, visible in visible_by_state.items():
        for group in exclusive:
            if len(set(group) & visible) > 1:
                raise ValueError(sid + ": mutually exclusive drawings are simultaneously visible")
    relations = _index(project.get("relations", []), "relations")
    frozen = "critical_relation_ids" in project
    inventory = _ids(project.get("critical_relation_ids", []), "critical_relation_ids")
    # Never silently omit a newly recorded critical boundary from the denominator.
    inventory |= {identity for identity, r in relations.items() if r.get("critical", True)}
    missing = sorted(inventory - relations.keys())
    incomplete, unknown, units, checked_order = [], [], [], []
    for identity, rel in relations.items():
        front, back = rel.get("front"), rel.get("back")
        if front not in drawings or back not in drawings or front == back:
            raise ValueError(identity + ": front/back must identify different existing drawing segments")
        absent = []
        roi = rel.get("source_roi")
        if roi is None:
            absent.append("source_roi")
        else:
            _finite(roi, 4, identity + ".source_roi")
            x0, y0, x1, y1 = roi
            if not (x0 < x1 and y0 < y1) or (canvas and not (0 <= x0 < x1 <= canvas[0] and 0 <= y0 < y1 <= canvas[1])):
                raise ValueError(identity + ": source_roi must have positive area inside the source canvas")
        owner = rel.get("line_owner")
        if owner is None:
            absent.append("line_owner")
        elif owner not in (front, back, "unknown"):
            raise ValueError(identity + ": line_owner must be front, back or unknown")
        points = rel.get("connection_points")
        if points is None:
            absent.append("connection_points")
        else:
            for pid, point in _index(points, identity + ".connection_points").items():
                _finite(point.get("point"), 2, pid + ".point")
                if point.get("drawing_id", front) not in (front, back):
                    raise ValueError(pid + ": connection drawing_id must be a relation endpoint")
                x, y = point["point"]
                if canvas and not (0 <= x <= canvas[0] and 0 <= y <= canvas[1]):
                    raise ValueError(pid + ": connection point outside source canvas")
        hidden = rel.get("hidden_requirements")
        if not isinstance(hidden, list) or not hidden or any(not isinstance(s, str) or not s.strip() for s in hidden):
            absent.append("hidden_requirements")
        selected_states = _ids(rel.get("states", []), identity + ".states")
        selected_cases = _ids(rel.get("test_cases", []), identity + ".test_cases")
        if selected_states - all_states or selected_cases - cases.keys():
            raise ValueError(identity + ": unknown states/test_cases reference")
        if not selected_states:
            absent.append("states")
        if not selected_cases:
            absent.append("test_cases")
        for cid in selected_cases:
            if cases[cid].get("state_id", "default") not in selected_states:
                raise ValueError(identity + ": test case state must occur in relation.states")
        observation = rel.get("observation", {})
        if not isinstance(observation, dict):
            observation = {}
        if observation.get("status") not in ("observed", "unknown"):
            absent.append("observation")
        elif observation["status"] == "observed" and not str(observation.get("evidence", "")).strip():
            absent.append("observation.evidence")
        if owner == "unknown" or observation.get("status", "unknown") == "unknown":
            unknown.append(identity)
        if absent:
            incomplete.append({"id": identity, "missing_fields": absent})
        if identity in inventory:
            for cid in sorted(selected_cases) or [None]:
                units.append({"relation_id": identity, "test_case_id": cid, "status": "pending"})
        # Check drawing segments, never topologically sort logical objects.
        for sid in sorted(selected_states):
            if {front, back} <= visible_by_state[sid]:
                if (not planning or "order" in drawings[front] and "order" in drawings[back]):
                    if order[front] <= order[back]:
                        raise ValueError(f"{identity}/{sid}: front {front} must be drawn after back {back}")
                    checked_order.append({"relation_id": identity, "state_id": sid})
                else:
                    warnings.append(f"{identity}/{sid}: planned z order unavailable; set drawing_specs.order before assembly")
    for identity in missing:
        units.append({"relation_id": identity, "test_case_id": None, "status": "pending"})
    incomplete_ids = {item["id"] for item in incomplete}
    complete_count = len(inventory - set(missing) - incomplete_ids)
    return {"inventory_frozen": frozen, "critical_relation_ids": sorted(inventory),
            "recorded_complete": complete_count, "required_relations": len(inventory) if inventory else None,
            "recording_coverage": complete_count / len(inventory) if frozen and inventory else None,
            "missing_relation_ids": missing, "incomplete_records": incomplete,
            "unknown_relation_ids": sorted(unknown), "required_test_units": units,
            "test_coverage": None, "checked_drawing_order": checked_order, "warnings": warnings,
            "note": "Structural checks only; unknown and untested relations never receive visual passes."}


def _neighbors(project, identities):
    result = set(identities)
    for rel in project.get("relations", []):
        endpoints = {rel.get("front"), rel.get("back")} - {None}
        if endpoints & set(identities):
            result |= endpoints
    return result


def affected_drawings(before, after, changed_ids=()):
    """One-hop local interfaces, plus descendants of changed motion parents.

    Use both snapshots: deletion or migration still invalidates the former neighbors.
    Independent alpha evidence uses its own fingerprint and remains reusable.
    """
    old = {p["id"]: p for p in before.get("parts", [])}
    new = {p["id"]: p for p in after.get("parts", [])}
    changed = set(changed_ids) | {pid for pid in old.keys() | new.keys() if old.get(pid) != new.get(pid)}
    old_materials = _index(before.get("materials", []), "materials")
    new_materials = _index(after.get("materials", []), "materials")
    changed_materials = {mid for mid in old_materials.keys() | new_materials.keys()
                         if old_materials.get(mid) != new_materials.get(mid)}
    changed |= {p["id"] for p in [*old.values(), *new.values()] if p.get("material_id") in changed_materials}
    old_rel = {r["id"]: r for r in before.get("relations", [])}
    new_rel = {r["id"]: r for r in after.get("relations", [])}
    old_states = {s["id"]: s for s in before.get("states", [])}
    new_states = {s["id"]: s for s in after.get("states", [])}
    for sid in old_states.keys() | new_states.keys():
        if old_states.get(sid) != new_states.get(sid):
            # A removed explicit state also changes its dependent review contract.
            changed |= set(old_states.get(sid, {}).get("visible", []))
            changed |= set(new_states.get(sid, {}).get("visible", []))
    old_cases = {c["id"]: c for c in before.get("test_cases", [])}
    new_cases = {c["id"]: c for c in after.get("test_cases", [])}
    changed_cases = {cid for cid in old_cases.keys() | new_cases.keys() if old_cases.get(cid) != new_cases.get(cid)}
    for rel in list(old_rel.values()) + list(new_rel.values()):
        if set(rel.get("test_cases", [])) & changed_cases:
            changed |= {rel.get("front"), rel.get("back")} - {None}
    if before.get("exclusive_groups", []) != after.get("exclusive_groups", []):
        changed |= {pid for project in (before, after) for group in project.get("exclusive_groups", []) for pid in group}
    for rid in old_rel.keys() | new_rel.keys():
        if old_rel.get(rid) != new_rel.get(rid):
            for rel in (old_rel.get(rid, {}), new_rel.get(rid, {})):
                changed |= {rel.get("front"), rel.get("back")} - {None}
    result = _neighbors(before, changed) | _neighbors(after, changed)
    for project in (before, after):
        parts = project.get("parts", [])
        ancestor_ids = {p.get("logical_id", p["id"]) for p in parts if p["id"] in changed}
        while True:
            more = {p.get("logical_id", p["id"]) for p in parts if p.get("motion_parent") in ancestor_ids}
            if more <= ancestor_ids:
                break
            ancestor_ids |= more
        result |= {p["id"] for p in parts if p.get("logical_id", p["id"]) in ancestor_ids}
    return sorted(result)


def dependency_fingerprint(project, drawing_ids, asset_hashes, rig=None,
                           relation_ids=None, state_ids=None, evidence_kind="motion"):
    """Bind evidence to pixels, interfaces, relative order, state cases and motion ancestry.

    asset_hashes maps drawing IDs to CURRENT registered-file SHA256 (missing => None).
    The caller supplies all current hashes so neighbor edits invalidate interface evidence.
    relation_ids/state_ids restrict a particular check; None selects all relevant ones.
    """
    parts = _index(project.get("parts", []), "parts")
    selected = set(drawing_ids)
    if selected - parts.keys():
        raise ValueError("Fingerprint references unknown drawing IDs")
    payload = {"version": 1, "evidence_kind": evidence_kind,
               "source_sha256": asset_hashes.get("__source__"),
               "drawing_ids": sorted(selected)}
    if evidence_kind == "alpha":
        payload["assets"] = {pid: asset_hashes.get(pid) for pid in sorted(selected)}
        return canonical_hash(payload)
    relation_index = _index(project.get("relations", []), "relations")
    relevant = {rid: r for rid, r in relation_index.items()
                if {r.get("front"), r.get("back")} & selected}
    if relation_ids is not None:
        relevant = {rid: relation_index.get(rid) for rid in sorted(set(relation_ids))}
    involved = set(selected)
    for rel in relevant.values():
        if rel:
            involved |= {rel.get("front"), rel.get("back")} - {None}
    if involved - parts.keys():
        raise ValueError("Fingerprint relation has unknown endpoints")
    payload["assets"] = {pid: asset_hashes.get(pid) for pid in sorted(involved)}
    payload["relations"] = relevant
    # Preserve custom ownership/hidden-surface/anchor requirements without a closed whitelist.
    excluded = {"asset", "file", "path", "name", "order", "visible", "crop"}
    payload["parts"] = {pid: {key: value for key, value in parts[pid].items() if key not in excluded}
                        for pid in sorted(involved)}
    if "materials" in project:
        material_ids = {parts[pid].get("material_id") for pid in involved}
        payload["materials"] = {m["id"]: m for m in project["materials"] if m["id"] in material_ids}
    # Normalized relative order: changing an unrelated layer's numeric value is not a local change.
    ordered = sorted(enumerate(parts.values()), key=lambda pair: (pair[1].get("order", pair[0]), pair[0]))
    payload["relative_order"] = [p["id"] for _, p in ordered if p["id"] in involved]
    states = _index(project.get("states", []), "states")
    selected_states = set(state_ids) if state_ids is not None else {"default", *states}
    payload["states"] = {sid: sorted(state_visibility(project, sid) & involved) for sid in sorted(selected_states)}
    payload["exclusive_groups"] = sorted([sorted(set(group) & involved) for group in project.get("exclusive_groups", []) if set(group) & involved])
    cases = _index(project.get("test_cases", []), "test_cases")
    relevant_case_ids = {cid for rel in relevant.values() if rel for cid in rel.get("test_cases", [])}
    payload["test_cases"] = {cid: cases.get(cid) for cid in sorted(relevant_case_ids)}
    if evidence_kind == "motion":
        logical = {p.get("logical_id", p["id"]): p for p in parts.values()}
        parents = {identity: p.get("motion_parent") for identity, p in logical.items()}
        _check_chains(parents, "motion_parent")
        ancestor_ids = set()
        for pid in involved:
            identity = parts[pid].get("logical_id", pid)
            while identity is not None:
                ancestor_ids.add(identity)
                identity = parents[identity]
        payload["motion_parents"] = {lid: parents[lid] for lid in sorted(ancestor_ids)}
        payload["motion_ancestor_parts"] = {
            pid: {key: value for key, value in part.items() if key not in excluded}
            for pid, part in sorted(parts.items()) if part.get("logical_id", pid) in ancestor_ids}
        if rig is not None:
            nodes = rig.get("nodes", {})
            mapping = rig.get("part_nodes", {})
            _check_chains({nid: node.get("parent") for nid, node in nodes.items()}, "rig")
            node_ids = set()
            for pid in involved:
                node_id = mapping.get(pid)
                while node_id is not None:
                    if node_id not in nodes:
                        raise ValueError("Unknown rig node " + str(node_id))
                    node_ids.add(node_id)
                    node_id = nodes[node_id].get("parent")
            payload["rig"] = {"nodes": {nid: nodes[nid] for nid in sorted(node_ids)},
                              "part_nodes": {pid: mapping.get(pid, "unmapped") for pid in sorted(involved)},
                              "canvas": rig.get("canvas"),
                              "order": [pid for pid in rig.get("order", []) if pid in involved]}
    return canonical_hash(payload)
