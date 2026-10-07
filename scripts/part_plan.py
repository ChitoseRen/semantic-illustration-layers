# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 ChitoseRen
"""Render an input-specific semantic plan and check its user-confirmation binding."""
from __future__ import annotations
import argparse
import base64
import hashlib
import html
import json
import math
from pathlib import Path
from PIL import Image
from relations import validate_relations


def digest(data):
    return hashlib.sha256(data).hexdigest()


def nonempty(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def simple_polygon(points):
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def on_segment(a, b, c):
        return (min(a[0],b[0])-1e-9 <= c[0] <= max(a[0],b[0])+1e-9
                and min(a[1],b[1])-1e-9 <= c[1] <= max(a[1],b[1])+1e-9)
    edges = list(zip(points, points[1:]+points[:1]))
    if any(a == b for a, b in edges):
        return False
    for i, (a, b) in enumerate(edges):
        for j in range(i+1, len(edges)):
            if j == i+1 or (i == 0 and j == len(edges)-1):
                continue
            c, d = edges[j]
            p, q, r, s = cross(a,b,c), cross(a,b,d), cross(c,d,a), cross(c,d,b)
            if p*q < 0 and r*s < 0:
                return False
            if ((abs(p)<1e-9 and on_segment(a,b,c)) or (abs(q)<1e-9 and on_segment(a,b,d))
                or (abs(r)<1e-9 and on_segment(c,d,a)) or (abs(s)<1e-9 and on_segment(c,d,b))):
                return False
    return True


def load_plan(path):
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError("Plan must be an object")
    source_relative = Path(nonempty(raw.get("source"), "source"))
    if source_relative.is_absolute() or source_relative.drive or source_relative.root:
        raise ValueError("source must be relative to plan.json")
    source = (path.parent / source_relative).resolve()
    data = source.read_bytes()
    with Image.open(source) as im:
        width, height = im.size
        mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}.get(im.format)
        if mime is None:
            raise ValueError("Source must be PNG, JPEG or WebP")
    nonempty(raw.get("purpose"), "purpose")
    nonempty(raw.get("motion_scope"), "motion_scope")
    parts = raw.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("parts must be a nonempty list")
    ids, drawing_ids = set(), set()
    normalized = []
    for number, part in enumerate(parts, 1):
        if not isinstance(part, dict):
            raise ValueError("Each part must be an object")
        identity = nonempty(part.get("id"), "part.id")
        if identity in ids:
            raise ValueError(f"Duplicate part id: {identity}")
        ids.add(identity)
        for key in ("name", "purpose"):
            nonempty(part.get(key), f"{identity}.{key}")
        drawings = part.get("drawings", [identity])
        if not isinstance(drawings, list) or not drawings:
            raise ValueError(f"{identity}.drawings must be nonempty")
        for drawing in drawings:
            nonempty(drawing, "drawing id")
            if drawing in drawing_ids:
                raise ValueError(f"Duplicate drawing id: {drawing}")
            drawing_ids.add(drawing)
        regions = part.get("regions")
        if not isinstance(regions, list) or not regions:
            raise ValueError(f"{identity}.regions must be nonempty")
        polygons = []
        for region in regions:
            if not isinstance(region, dict) or len(region) != 1:
                raise ValueError("Region must have exactly one box or polygon")
            if "box" in region:
                box = region["box"]
                if not isinstance(box, list) or len(box) != 4:
                    raise ValueError("box must contain four coordinates")
                x0, y0, x1, y1 = box
                points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            else:
                points = region.get("polygon")
            if not isinstance(points, list) or len(points) < 3:
                raise ValueError("polygon must have at least three points")
            for point in points:
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError("A point must have two coordinates")
                if any(isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) for n in point):
                    raise ValueError("Coordinates must be finite numbers")
                if not 0 <= point[0] <= width or not 0 <= point[1] <= height:
                    raise ValueError("Region extends outside source canvas")
            if "box" in region and not (x0 < x1 and y0 < y1):
                raise ValueError("Box has no positive area")
            if len(points)>3 and points[-1] == points[0]:
                points = points[:-1]
            if not simple_polygon(points):
                raise ValueError("Polygon self-intersects or repeats an edge point")
            area = sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(points, points[1:]+points[:1]))
            if abs(area) < 1e-8:
                raise ValueError("Polygon has no area")
            polygons.append(points)
        normalized.append(dict(part, number=number, drawings=drawings, polygons=polygons))
    relation_report = validate_relations(raw, [width, height], planning=True)
    questions = raw.get("questions", [])
    if not isinstance(questions, list) or any(not isinstance(q, str) for q in questions):
        raise ValueError("questions must be a list of strings")
    source_sha = digest(data)
    canonical = {k: v for k, v in raw.items() if k not in ("confirmation", "source")}
    canonical["source_sha256"] = source_sha
    plan_sha = digest(json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    confirmation = raw.get("confirmation") or {}
    valid = (isinstance(confirmation, dict)
             and confirmation.get("status") in ("confirmed", "waived")
             and confirmation.get("plan_sha256") == plan_sha
             and confirmation.get("source_sha256") == source_sha
             and isinstance(confirmation.get("user_message"), str)
             and bool(confirmation["user_message"].strip()))
    proposal = dict(source_sha256=source_sha, plan_sha256=plan_sha,
                    state=confirmation["status"] if valid else "waiting_for_confirmation",
                    confirmation_valid=valid, logical_part_count=len(parts), drawing_count=len(drawing_ids),
                    canvas=[width, height], purpose=raw["purpose"], motion_scope=raw["motion_scope"],
                    parts=parts, questions=questions, relations=raw.get("relations", []),
                    materials=raw.get("materials", []),
                    relation_validation=relation_report,
                    note="Diagram only; no extraction or generated pixels. User confirmation is distinct from visual QA.")
    return raw, normalized, proposal, data, mime


CSS = """
*{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:#202a37;font:15px/1.6 system-ui,'Microsoft YaHei',sans-serif}
main{max-width:1440px;margin:auto;padding:28px}h1{margin:0 0 8px;font-size:28px}p{margin:8px 0}
.badge{display:inline-block;background:#fff2cf;color:#745412;padding:5px 12px;border-radius:8px}
.grid{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(300px,1fr);gap:22px;margin-top:22px}
.figure{position:sticky;top:12px;align-self:start;background:white;border-radius:12px;padding:12px}
svg{display:block;width:100%;height:auto}.region{fill-opacity:.10;stroke-width:3;vector-effect:non-scaling-stroke}
.mark{transition:opacity .12s}.mark.dim{opacity:.12}.mark.active .region{fill-opacity:.28;stroke-width:5}
.label{fill:#fff;stroke:#202a37;stroke-width:3;paint-order:stroke;font-weight:800;text-anchor:middle;dominant-baseline:middle}
button{font:inherit;cursor:pointer}.reset{background:white;border:1px solid #b7c3d2;border-radius:7px;padding:7px 12px}
.part{display:block;width:100%;text-align:left;background:white;border:2px solid transparent;border-radius:10px;padding:12px 15px;margin-bottom:10px}
.part[aria-pressed=true]{border-color:#275dc6;box-shadow:0 0 0 2px #d8e4ff}
.title{display:block;font-weight:700}.detail{display:block;color:#4c5969;font-size:13px;margin-top:5px}
.num{display:inline-grid;place-items:center;min-width:26px;height:26px;border-radius:50%;color:white;margin-right:8px}
.note{font-size:13px;color:#526172}.questions{background:#fff;padding:16px;border-radius:10px;margin:18px 0}
.relations{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:16px;margin:18px 0}.relation{background:#fff;padding:16px;border-radius:10px}.relation svg{max-height:360px;background:#e8edf3;image-rendering:pixelated}.relation .unknown{color:#9b5500}.connection{fill:#ffeb3b;stroke:#202a37;stroke-width:1;vector-effect:non-scaling-stroke}
@media(max-width:760px){main{padding:16px}.grid{grid-template-columns:1fr}.figure{position:static}h1{font-size:24px}}
"""
JS = """
const rows=Array.from(document.querySelectorAll('.part'));
const marks=Array.from(document.querySelectorAll('.mark'));
function select(n){rows.forEach((r,i)=>r.setAttribute('aria-pressed',String(i===n)));
marks.forEach((m,i)=>{m.classList.toggle('active',i===n);m.classList.toggle('dim',n>=0&&i!==n);});}
rows.forEach((r,i)=>r.addEventListener('click',()=>select(r.getAttribute('aria-pressed')==='true'?-1:i)));
document.querySelector('.reset').addEventListener('click',()=>select(-1));
"""


def render(raw, parts, proposal, data, mime):
    esc = lambda s: html.escape(str(s), quote=True)
    width, height = proposal["canvas"]
    font_size = max(width, height)*.023
    shapes, buttons = [], []
    for i, part in enumerate(parts):
        colour = f"hsl({i*137.508%360:.1f} 66% 43%)"
        polygons = "".join('<polygon class="region" points="'+
                           " ".join(f"{x},{y}" for x, y in points)+
                           f'" fill="{colour}" stroke="{colour}"/>' for points in part["polygons"])
        points = part["polygons"][0]
        cx = sum(p[0] for p in points)/len(points)
        cy = sum(p[1] for p in points)/len(points)
        shapes.append(f'<g class="mark"><title>{esc(part["name"])}</title>{polygons}'
                      f'<text class="label" x="{cx}" y="{cy}" font-size="{font_size}">{i+1}</text></g>')
        buttons.append(f'<button type="button" class="part" aria-pressed="false">'
                       f'<span class="title"><span class="num" style="background:{colour}">{i+1}</span>{esc(part["name"])}</span>'
                       f'<span class="detail">{esc(part["purpose"])}</span>'
                       f'<span class="detail">{len(part["drawings"])} 个绘制层或表情附件</span>'
                       f'<span class="detail">语义归属：{esc(part.get("semantic_parent",part.get("parent","无")))}；运动继承：{esc(part.get("motion_parent","未声明"))}；制作依赖：{esc(", ".join(part.get("depends_on",[])) or "无")}</span>'
                       f'<span class="detail">{esc(part.get("depth_notes",""))}</span></button>')
    state_text = {"waiting_for_confirmation":"待用户确认", "confirmed":"已记录用户确认", "waived":"已记录明确免确认"}[proposal["state"]]
    questions = "".join("<li>"+esc(q)+"</li>" for q in proposal["questions"])
    data_uri = f"data:{mime};base64,"+base64.b64encode(data).decode("ascii")
    relation_cards, connection_marks = [], []
    for rel in raw.get("relations", []):
        roi = rel.get("source_roi")
        markers = []
        for point in rel.get("connection_points", []):
            x, y = point["point"]
            marker = (f'<circle class="connection" cx="{x}" cy="{y}" r="{max(1.0, max(width,height)*.003)}">'
                      f'<title>{esc(rel["id"])} / {esc(point["id"])}</title></circle>')
            connection_marks.append(marker)
            radius = max(0.5, min(roi[2]-roi[0], roi[3]-roi[1])*.012) if roi else 1
            markers.append(f'<circle class="connection" cx="{x}" cy="{y}" r="{radius}"><title>{esc(point["id"])}</title></circle>')
        zoom = '<p class="unknown">未标定源图 ROI</p>'
        if roi:
            x0, y0, x1, y1 = roi
            zoom = (f'<svg viewBox="{x0} {y0} {x1-x0} {y1-y0}" role="img" aria-label="{esc(rel["id"])} 局部关系放大">'
                    f'<image href="{data_uri}" width="{width}" height="{height}"/>'+''.join(markers)+'</svg>')
        status = rel.get("observation", {}).get("status", "unknown")
        evidence = rel.get("observation", {}).get("evidence", "未观察")
        relation_cards.append('<article class="relation"><strong>'+esc(rel["id"])+ '</strong>'+zoom
            +f'<p>{esc(rel["front"])} 在前 → {esc(rel["back"])} 在后</p>'
            +f'<p>线条归属：{esc(rel.get("line_owner","unknown"))}</p>'
            +f'<p>隐藏结构：{esc("；".join(rel.get("hidden_requirements",[])) or "未记录")}</p>'
            +f'<p class="note">状态：{esc(", ".join(rel.get("states",[])) or "未登记")}；用例：{esc(", ".join(rel.get("test_cases",[])) or "未登记")}</p>'
            +f'<p class="note {"unknown" if status == "unknown" else ""}">观察：{esc(status)}；{esc(evidence)}</p></article>')
    relation_report = proposal["relation_validation"]
    coverage = relation_report["recording_coverage"]
    coverage_text = "未建立冻结清单，覆盖率不可用" if coverage is None else f'记录完整 {relation_report["recorded_complete"]}/{relation_report["required_relations"]}（{coverage:.0%}）'
    relation_summary = (f'<h2>关键局部关系与连接点</h2><p>{esc(coverage_text)}；这不是视觉或动作通过率。</p>'
                        f'<p class="note">缺记录：{esc(", ".join(relation_report["missing_relation_ids"]) or "无")}；观察未知：{esc(", ".join(relation_report["unknown_relation_ids"]) or "无")}</p>')
    material_cards = []
    for material in raw.get("materials", []):
        details = []
        for part in parts:
            for pid in part["drawings"]:
                spec = {**part, **raw.get("drawing_specs", {}).get(pid, {})}
                if spec.get("material_id") == material["id"]:
                    details.append(f'<li>{esc(pid)} · {esc(spec.get("paint_role"))}'
                                   f' · 接收面：{esc(spec.get("receiver_id", "自身"))}'
                                   f' · 投射者：{esc(", ".join(spec.get("caster_ids", [])) or spec.get("caster_note", "无"))}'
                                   f' · 随动：{esc(spec.get("motion_policy", spec.get("motion_parent", "未声明")))}</li>')
        material_cards.append('<article class="relation"><strong>'+esc(material["name"])+ '</strong>'
            +f'<p>固有色：{esc(material["base_color_note"])}</p><p>源图依据：{esc(material["source_evidence"])}</p>'
            +f'<p>可移动底色层：{esc(", ".join(material["base_drawings"]))}</p><ul>'+''.join(details)+'</ul></article>')
    material_summary = ('<h2>固有色与光影分层</h2><section class="relations">'+''.join(material_cards)+'</section>'
                        if material_cards else '<p class="note">未登记固有色分层；旧方案可读取，此状态不代表全图材质已检查。</p>')
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>语义分区确认示意</title><style>'+CSS+'</style><main>'
            '<h1>语义分区确认示意</h1><span class="badge">'+state_text+'</span>'
            f'<p><strong>{len(parts)} 个逻辑部件 · {proposal["drawing_count"]} 个绘制段/附件</strong></p>'
            f'<p>{esc(raw["purpose"])}</p><p class="note">动作范围：{esc(raw["motion_scope"])}</p>'
            '<p class="note">示意区域不等于最终抠图边界。点击部件清单可突出对应编号，分区确认不代表素材质量已经通过。</p>'
            '<div class="grid"><section class="figure"><button class="reset">显示全部区域</button>'
            f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="原图上的编号分区示意">'
            f'<image href="{data_uri}" width="{width}" height="{height}"/>'
            +''.join(shapes)+''.join(connection_marks)+'</svg></section><section aria-label="部件清单">'+''.join(buttons)+'</section></div>'
            +material_summary+relation_summary+'<section class="relations">'+''.join(relation_cards)+'</section>'
            '<section class="questions"><strong>请在对话中确认或指出调整</strong><ul>'+questions+'</ul>'
            '<p>可以直接提出合并、细分或变更活动部位。页面操作只用于查看，不会自动批准方案。</p></section>'
            '</main><script>'+JS+'</script></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check-confirmation", action="store_true")
    args = parser.parse_args()
    if not args.check_confirmation and args.out is None:
        parser.error("--out is required when rendering")
    try:
        raw, parts, proposal, data, mime = load_plan(args.plan.resolve())
        if args.check_confirmation:
            print(json.dumps({k:proposal[k] for k in ("state","confirmation_valid","plan_sha256","source_sha256")}))
            return 0 if proposal["confirmation_valid"] else 2
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out/"index.html").write_text(render(raw, parts, proposal, data, mime), encoding="utf-8")
        (args.out/"proposal.json").write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"state":proposal["state"], "logical_parts":len(parts), "drawings":proposal["drawing_count"],
                          "html":str((args.out/"index.html").resolve())}, ensure_ascii=False))
        return 0
    except (ValueError, OSError, TypeError, KeyError) as error:
        print(json.dumps({"error":str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
