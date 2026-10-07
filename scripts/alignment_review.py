# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 ChitoseRen
"""Create read-only, same-canvas geometry diagnostics and a standalone ROI viewer.

Pillow and numpy only. No alignment transform is fitted or applied. Numerical
agreement never constitutes visual acceptance. CLI: alignment_review.py config.json
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image


MAX_PIXELS = 20_000_000
MAX_ROI_PIXELS = 12_000_000
MAX_DIMENSION = 16_384
MAX_POINTS = 512
MAX_REGIONS = 64
MAX_JSON_BYTES = 2_000_000
NEUTRAL_RGB = (128, 128, 128)
READY_GEOMETRY = "geometry_within_limit_needs_visual_review"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("JSON exceeds 2 MB")
    value = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def finite_number(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def nonempty_text(value, field, maximum=256):
    if (not isinstance(value, str) or not value.strip() or len(value) > maximum
            or any(ord(character) < 32 for character in value)):
        raise ValueError(f"{field}: require nonempty text of at most {maximum} characters without control characters")
    return value


def check_hash(value, field):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise ValueError(f"{field}: require a 64-character SHA256")
    return value.lower()


def inside(root, value, field, *, relative_only=False, must_exist=True):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{field}: path is required")
    path = Path(value)
    if relative_only and (path.is_absolute() or path.drive):
        raise ValueError(f"{field}: path must be relative to root")
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"{field}: path must remain inside root")
    if must_exist and not path.is_file():
        raise ValueError(f"{field}: input file does not exist")
    return path


def relative(root, path):
    return Path(path).relative_to(root).as_posix()


def read_composite(path, field):
    with Image.open(path) as image:
        width, height = image.size
        if min(width, height) < 1 or width * height > MAX_PIXELS or max(width, height) > MAX_DIMENSION:
            raise ValueError(f"{field}: unsupported canvas bounds")
        if image.mode not in ("RGB", "RGBA"):
            raise ValueError(f"{field}: require RGB or RGBA")
        if image.info.get("icc_profile"):
            raise ValueError(f"{field}: embedded ICC profile; explicitly normalize color interpretation to sRGB first")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"{field}: multi-frame input is not supported")
        rgba = image.convert("RGBA")
        has_transparency = rgba.getchannel("A").getextrema()[0] < 255
        # Each input independently receives the SAME neutral backing, never the source image.
        background = Image.new("RGBA", image.size, (*NEUTRAL_RGB, 255))
        display = Image.alpha_composite(background, rgba).convert("RGB")
    return display, has_transparency


def validate_regions(settings, size):
    regions = settings.get("regions")
    if not isinstance(regions, list) or not 1 <= len(regions) <= MAX_REGIONS:
        raise ValueError(f"regions: require 1..{MAX_REGIONS} explicit ROI records")
    identifiers, total_pixels = set(), 0
    for region in regions:
        if not isinstance(region, dict):
            raise ValueError("regions: each ROI must be an object")
        identifier = nonempty_text(region.get("id"), "regions.id", 128)
        if identifier in identifiers:
            raise ValueError("regions: duplicate id")
        identifiers.add(identifier)
        box = region.get("box")
        if (not isinstance(box, list) or len(box) != 4
                or any(not isinstance(v, int) or isinstance(v, bool) for v in box)):
            raise ValueError("regions.box: require four integer coordinates [left,top,right,bottom]")
        left, top, right, bottom = box
        if not 0 <= left < right <= size[0] or not 0 <= top < bottom <= size[1]:
            raise ValueError("regions.box: ROI is empty or outside the source canvas")
        total_pixels += (right - left) * (bottom - top)
    if total_pixels > MAX_ROI_PIXELS:
        raise ValueError("regions: total ROI pixels exceed standalone viewer budget (12 million)")
    return regions


def validate_point_file(path, kind, size):
    document = read_json(path)
    hash_field = "source_sha256" if kind == "reference" else "candidate_sha256"
    declared_hash = check_hash(document.get(hash_field), f"{kind}.{hash_field}")
    points = document.get("points")
    if not isinstance(points, list) or len(points) > MAX_POINTS:
        raise ValueError(f"{kind}.points: require a list with at most {MAX_POINTS} points")
    lookup = {}
    for point in points:
        if not isinstance(point, dict):
            raise ValueError(f"{kind}.points: each point must be an object")
        identifier = nonempty_text(point.get("id"), f"{kind}.id", 128)
        if identifier in lookup:
            raise ValueError(f"{kind}: duplicate point id {identifier}")
        xy = point.get("xy")
        if (not isinstance(xy, list) or len(xy) != 2 or any(not finite_number(v) for v in xy)
                or not 0 <= xy[0] <= size[0] - 1 or not 0 <= xy[1] <= size[1] - 1):
            raise ValueError(f"{kind}.{identifier}: xy must be finite source-canvas pixel-center coordinates")
        item = {"id": identifier, "xy": xy}
        if kind == "reference":
            role = point.get("role")
            if role not in ("fit", "check"):
                raise ValueError(f"reference.{identifier}: role must be fit or check")
            item["role"] = role
            if "part" in point:
                item["part"] = nonempty_text(point["part"], f"reference.{identifier}.part", 128)
        else:
            item["evidence"] = nonempty_text(point.get("evidence"), f"observed.{identifier}.evidence", 4000)
        lookup[identifier] = item
    return declared_hash, lookup


def validate_pairs(settings, reference):
    pairs = settings.get("pairs", [])
    if not isinstance(pairs, list) or len(pairs) > MAX_POINTS:
        raise ValueError("pairs: require at most 512 pair records")
    seen = set()
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError("pairs: each pair must be an object")
        identifier = nonempty_text(pair.get("id"), "pairs.id", 128)
        if identifier in seen:
            raise ValueError("pairs: duplicate id")
        seen.add(identifier)
        a = nonempty_text(pair.get("a"), "pairs.a", 128)
        b = nonempty_text(pair.get("b"), "pairs.b", 128)
        if a == b:
            raise ValueError("pairs: endpoints must use different point IDs")
        if reference is not None and (a not in reference or b not in reference):
            raise ValueError(f"pairs.{identifier}: endpoint refers to an unknown reference point")
    return pairs


def pair_diagnostic(pair, reference, observed, trusted):
    result = {"id": pair["id"], "a": pair["a"], "b": pair["b"], "status": "unavailable", "counts_toward_acceptance": False}
    if not trusted or not reference or not observed or any(p not in observed for p in (pair["a"], pair["b"])):
        result["reason"] = "Missing or stale reference/observations"
        return result
    a, b = pair["a"], pair["b"]
    source_vector = np.array(reference[b]["xy"], dtype=float) - reference[a]["xy"]
    candidate_vector = np.array(observed[b]["xy"], dtype=float) - observed[a]["xy"]
    source_length = float(np.linalg.norm(source_vector))
    candidate_length = float(np.linalg.norm(candidate_vector))
    source_angle = math.degrees(math.atan2(source_vector[1], source_vector[0])) if source_length else None
    candidate_angle = math.degrees(math.atan2(candidate_vector[1], candidate_vector[0])) if candidate_length else None
    result.update(status="diagnostic", source_vector_px=source_vector.tolist(),
                  candidate_vector_px=candidate_vector.tolist(), source_length_px=source_length,
                  candidate_length_px=candidate_length, length_difference_px=candidate_length - source_length,
                  source_angle_degrees=source_angle, candidate_angle_degrees=candidate_angle,
                  angle_difference_degrees=((candidate_angle - source_angle + 180) % 360 - 180)
                  if source_angle is not None and candidate_angle is not None else None,
                  angle_convention="atan2(dy,dx), y down; clockwise positive; signed difference in [-180,180)")
    return result


def measure_geometry(reference, observed, reference_fresh, observed_fresh, limit):
    reference = reference or {}
    observed = observed or {}
    trusted = reference_fresh and observed_fresh
    fit = [point for point in reference.values() if point["role"] == "fit"]
    checks = [point for point in reference.values() if point["role"] == "check"]
    excluded, rows = [], []
    for point in reference.values():
        shared_fit = [other["id"] for other in fit
                      if point["role"] == "check" and other.get("part") == point.get("part")
                      and other["xy"] == point["xy"]]
        if shared_fit:
            excluded.append({"id": point["id"], "reason": "Same source coordinate and part as a fit point", "fit_ids": shared_fit})
        observation = observed.get(point["id"])
        row = {"id": point["id"], "role": point["role"], "part": point.get("part"),
               "source_xy": point["xy"], "candidate_xy": observation["xy"] if observation else None,
               "evidence": observation["evidence"] if observation else None,
               "independent_check": point["role"] == "check" and not shared_fit,
               "dx_px": None, "dy_px": None, "error_px": None,
               "status": "unavailable", "counts_toward_geometry": False}
        if trusted and observation:
            difference = np.array(observation["xy"], dtype=float) - point["xy"]
            row.update(dx_px=float(difference[0]), dy_px=float(difference[1]),
                       error_px=float(np.linalg.norm(difference)), status="diagnostic",
                       counts_toward_geometry=row["independent_check"])
        rows.append(row)
    eligible = [row for row in rows if row["independent_check"]]
    missing = [point["id"] for point in checks if point["id"] not in observed]
    errors = [row["error_px"] for row in eligible if row["error_px"] is not None]
    maximum = max(errors) if errors else None
    if not reference_fresh or not reference:
        status = "pending_reference"
    elif not eligible:
        status = "pending_independent_checks"
    elif not observed_fresh or missing:
        status = "pending_observations"
    elif limit is None:
        status = "pending_limit"
    elif maximum is not None and maximum > limit:
        status = "geometry_failed"
    else:
        status = READY_GEOMETRY
    return {"status": status, "threshold_px": limit, "maximum_check_error_px": maximum,
            "declared_check_count": len(checks), "independent_check_count": len(eligible),
            "measured_independent_check_count": len(errors), "missing_check_ids": missing,
            "excluded_checks": excluded, "checks": [row for row in rows if row["role"] == "check"],
            "fit_points": [row for row in rows if row["role"] == "fit"],
            "fit_points_count_toward_acceptance": False,
            "transform_applied": {"matrix": [[1, 0, 0], [0, 1, 0]], "description": "Identity only; no fitting, registration, translation or resampling"}}


def gradient(rgb):
    values = np.asarray(rgb, dtype=np.float32) @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    padded = np.pad(values, 1, mode="edge")
    dx = (padded[1:-1, 2:] - padded[1:-1, :-2]) * 0.5
    dy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) * 0.5
    return np.clip(np.hypot(dx, dy) * 4, 0, 255).astype(np.uint8)


def data_url(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def make_roi(region, index, source, candidate, root, output):
    box = tuple(region["box"])
    source_crop, candidate_crop = source.crop(box), candidate.crop(box)
    width, height = source_crop.size
    side_by_side = Image.new("RGB", (width * 2, height))
    side_by_side.paste(source_crop, (0, 0))
    side_by_side.paste(candidate_crop, (width, 0))
    difference = np.abs(np.asarray(candidate_crop, dtype=np.int16) - np.asarray(source_crop, dtype=np.int16))
    difference_image = Image.fromarray(np.clip(difference * 4, 0, 255).astype(np.uint8), "RGB")
    # Calculate gradients before ROI cropping so crop borders cannot fabricate edges.
    margin = (max(0, box[0] - 1), max(0, box[1] - 1), min(source.width, box[2] + 1), min(source.height, box[3] + 1))
    inner = (box[0] - margin[0], box[1] - margin[1], box[2] - margin[0], box[3] - margin[1])
    source_edges = Image.fromarray(gradient(source.crop(margin)), "L").crop(inner)
    candidate_edges = Image.fromarray(gradient(candidate.crop(margin)), "L").crop(inner)
    edge_image = Image.merge("RGB", (source_edges, candidate_edges, candidate_edges))
    images = {"source": source_crop, "candidate": candidate_crop,
              "side_by_side": side_by_side, "overlay_50": Image.blend(source_crop, candidate_crop, 0.5),
              "rgb_difference": difference_image, "red_cyan_gradient": edge_image}
    artifacts = {}
    for name, image in images.items():
        path = output / f"roi-{index:03d}-{name}.png"
        image.save(path)
        artifacts[name] = relative(root, path)
    result = {"id": region["id"], "box": region["box"], "size": [width, height], "artifacts": artifacts,
              "side_by_side_order": "left=source, right=candidate; 1:1 source pixels",
              "rgb_difference": "absolute encoded sRGB channel difference x4, clipped; not geometric distance",
              "red_cyan_gradient": "red=source gradient, cyan=candidate gradient, white=overlap; intensity is not contour distance"}
    viewer = {"id": region["id"], "box": region["box"], "width": width, "height": height,
              "images": {name: data_url(images[name]) for name in ("source", "candidate", "rgb_difference", "red_cyan_gradient")}}
    return result, viewer


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>同画布对齐复核</title><style>
:root{font-family:system-ui,"Microsoft YaHei",sans-serif;color:#172134;background:#eef1f5}body{margin:0;padding:20px}
h1{font-size:23px;margin:0 0 12px}h2{font-size:17px}.card{background:white;border:1px solid #d3dae4;border-radius:9px;padding:16px;margin-bottom:14px}
.hint{color:#4b596c;font-size:13px;line-height:1.65}button,select{padding:8px 11px;border:1px solid #aebbc9;border-radius:5px;background:white;color:#172134;cursor:pointer}
button.active{background:#174f83;color:white}button:focus-visible,select:focus-visible{outline:3px solid #70b8ed}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:9px;margin:12px 0}
#viewport{overflow:auto;max-height:75vh;background:#b7bec8;border:1px solid #8797aa;padding:12px}canvas{display:block;image-rendering:pixelated;max-width:none;max-height:none}
#sample{font-variant-numeric:tabular-nums;min-height:22px}.status{font-weight:650;color:#9b3b14}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #dce2ea;text-align:left;vertical-align:top}td{word-break:break-word}.table-scroll{overflow:auto}code{word-break:break-all}details{margin-top:12px}label{font-size:14px}
</style></head><body>
<h1>同画布对齐复核</h1>
<div class="card"><div id="status" class="status"></div><p class="hint">所有视图采用同一个原画布 ROI；只改变显示倍数，不执行平移、旋转、缩放配准或自动对齐。左上像素中心为 (0,0)。坐标小数表示计算精度；人工标出约 0.5px 的差异，不证明形体达到亚像素完美。数值达标后仍需实际查看叠图，不能作为视觉通过。</p>
<div id="summary" class="hint"></div><details><summary>输入版本与检查指纹</summary><div id="hashes" class="hint"></div></details></div>
<div class="card"><div class="controls"><label>检查区域 <select id="region"></select></label><label>显示倍数 <select id="zoom"><option value="1">1 倍</option><option value="4">4 倍</option><option value="8">8 倍</option></select></label><label><input type="checkbox" id="markers"> 显示既有地标</label><label><input type="checkbox" id="markerLabels"> 显示地标名称（需先显示地标）</label></div>
<div class="controls" id="modes"><button data-mode="source">原图</button><button data-mode="candidate">当前组装</button><button data-mode="overlay" class="active">透明叠加</button><button data-mode="side_by_side">并排</button><button data-mode="rgb_difference">RGB 差异</button><button data-mode="red_cyan_gradient">红青梯度</button><button id="blink">开始 500ms 闪烁</button></div>
<div class="controls"><label>当前组装叠加透明度 <input id="opacity" type="range" min="0" max="100" value="50"> <span id="opacityValue">50%</span></label></div>
<p id="coordinates" class="hint"></p><p id="legend" class="hint"></p><div id="viewport"><canvas id="canvas"></canvas></div><p id="sample" class="hint">移动鼠标可查看原画布坐标。这里仅作只读比较，不产生观测点或验收意见。</p></div>
<div class="card"><h2>独立核验点</h2><div id="checks" class="table-scroll"></div><h2>拟合点（单列，不参与核验结论）</h2><div id="fits" class="table-scroll"></div><h2>点对派生诊断（不额外计票）</h2><div id="pairs" class="table-scroll"></div></div>
<script>
const DATA=__DATA__;
const $=id=>document.getElementById(id);
const statusLabels={pending_reference:'待补充或更新原图标定',pending_observations:'待补充或更新当前组装观测',pending_independent_checks:'缺少独立核验点',pending_limit:'尚未声明像素误差阈值',geometry_failed:'几何误差超过阈值',geometry_within_limit_needs_visual_review:'几何量测在阈值内，仍需实际看图'};
$('status').textContent=statusLabels[DATA.status]||DATA.status;
$('summary').textContent=`原画布 ${DATA.canvas[0]} × ${DATA.canvas[1]}；原图与当前图分别叠在 RGB(128,128,128) 中性底。透明缺口不会用原图补齐。核验点最大误差：${DATA.geometry.maximum_check_error_px??'未量测'} px；阈值：${DATA.geometry.threshold_px??'未声明'} px。`;
for(const [name,value] of Object.entries(DATA.hashes)){const p=document.createElement('p');p.textContent=name+': '+(typeof value==='object'?JSON.stringify(value):value);$('hashes').append(p)}
const receipt=document.createElement('p');receipt.textContent='检查指纹：'+DATA.fingerprint;$('hashes').append(receipt);
function table(target,headers,rows){if(!rows.length){$(target).textContent='暂无可用记录';return}const t=document.createElement('table');const head=document.createElement('tr');for(const h of headers){const c=document.createElement('th');c.textContent=h;head.append(c)}t.append(head);for(const row of rows){const tr=document.createElement('tr');for(const value of row){const td=document.createElement('td');td.textContent=value==null?'—':String(value);tr.append(td)}t.append(tr)}$(target).append(t)}
const show=v=>v==null?'—':(Array.isArray(v)?v.map(n=>Number(n).toFixed(3)).join(', '):Number(v).toFixed(3));
function pointRows(rows){return rows.map(p=>[p.id,p.part??'—',show(p.source_xy)+(p.source_xy&&DATA.hashes.reference.status==='stale'?'（过期存档）':''),show(p.candidate_xy)+(p.candidate_xy&&DATA.hashes.observed.status==='stale'?'（过期存档）':''),show(p.dx_px),show(p.dy_px),show(p.error_px),p.independent_check?'独立核验':'不计票',(p.evidence&&DATA.hashes.observed.status==='stale'?'过期存档，不代表当前文件：':'')+(p.evidence??'无候选观测证据')])}
table('checks',['ID','部件','原图 x,y','当前 x,y','dx px','dy px','误差 px','用途','观测证据'],pointRows(DATA.geometry.checks));
table('fits',['ID','部件','原图 x,y','当前 x,y','dx px','dy px','误差 px','用途','观测证据'],pointRows(DATA.geometry.fit_points));
table('pairs',['点对','源向量','当前向量','长度差 px','角度差 °','状态'],DATA.pairs.map(p=>[p.id,show(p.source_vector_px),show(p.candidate_vector_px),show(p.length_difference_px),show(p.angle_difference_degrees),p.status]));
for(let i=0;i<DATA.regions.length;i++){const o=document.createElement('option');o.value=i;o.textContent=DATA.regions[i].id;$('region').append(o)}
const cache=new Map();let region=DATA.regions[0],mode='overlay',blinkTimer=null,blinkMode='source';
async function loadImages(r){if(cache.has(r.id))return cache.get(r.id);const images={};await Promise.all(Object.entries(r.images).map(([key,url])=>new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>{images[key]=image;resolve()};image.onerror=reject;image.src=url})));cache.set(r.id,images);return images}
function stopBlink(){if(blinkTimer!==null)clearInterval(blinkTimer);blinkTimer=null;$('blink').textContent='开始 500ms 闪烁'}
function mark(ctx,point,observed,offset=0){if((observed?DATA.hashes.observed:DATA.hashes.reference).status!=='current')return;const xy=observed?point.candidate_xy:point.source_xy;if(!xy)return;const x=xy[0]-region.box[0],y=xy[1]-region.box[1];if(x<0||y<0||x>=region.width||y>=region.height)return;const px=x+0.5+offset,py=y+0.5;ctx.strokeStyle=observed?'#00ffff':'#ff3434';ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=1;ctx.beginPath();if(observed)ctx.arc(px,py,3,0,2*Math.PI);else{ctx.moveTo(px-4,py);ctx.lineTo(px+4,py);ctx.moveTo(px,py-4);ctx.lineTo(px,py+4)}ctx.stroke();if($('markerLabels').checked){ctx.font='10px sans-serif';ctx.fillText(point.id,px+5,py-4)}}
async function draw(){$('sample').textContent='移动鼠标可查看当前视图的原画布坐标。这里只作只读比较，不产生观测点或验收意见。';const requested=region;const images=await loadImages(requested);if(requested!==region)return;const view=blinkTimer===null?mode:blinkMode;const canvas=$('canvas');const doubled=view==='side_by_side';canvas.width=region.width*(doubled?2:1);canvas.height=region.height;canvas.style.width=canvas.width*Number($('zoom').value)+'px';canvas.style.height=canvas.height*Number($('zoom').value)+'px';const ctx=canvas.getContext('2d');ctx.imageSmoothingEnabled=false;ctx.globalAlpha=1;
if(view==='overlay'){ctx.drawImage(images.source,0,0);ctx.globalAlpha=Number($('opacity').value)/100;ctx.drawImage(images.candidate,0,0);ctx.globalAlpha=1}else if(doubled){ctx.drawImage(images.source,0,0);ctx.drawImage(images.candidate,region.width,0)}else{ctx.drawImage(images[view],0,0)}
if($('markers').checked){for(const point of [...DATA.geometry.fit_points,...DATA.geometry.checks]){if(doubled){mark(ctx,point,false);mark(ctx,point,true,region.width)}else{if(view!=='candidate')mark(ctx,point,false);if(view!=='source')mark(ctx,point,true)}}}
$('coordinates').textContent=`原画布 ROI [${region.box.join(', ')}]（右、下边界不包含），${region.width} × ${region.height} 原像素；左上原点，x 向右，y 向下；显示 ${$('zoom').value} 倍。`;
const descriptions={source:'原图；红色十字为原图标定点。',candidate:'当前组装；青色圆圈为当前文件的观测点。',overlay:'透明叠加：两图共享原始坐标，不自动修正偏移。红色为原图点，青色为当前点。',side_by_side:'左：原图；右：当前组装。两侧使用相同 ROI 和相同显示倍数。',rgb_difference:'RGB 绝对差 ×4，超出 255 截断。亮色可能来自颜色或位置变化，不能当作轮廓距离。',red_cyan_gradient:'红：原图梯度；青：当前梯度；白：重叠。梯度亮度受色彩影响，不等于轮廓距离。'};
const versionWarning=(DATA.hashes.reference.status==='stale'?' 原图标定已过期，仅保留在表格中存档，不在当前原图上绘制。':'')+(DATA.hashes.observed.status==='stale'?' 候选观测已过期，仅保留在表格中存档，不代表当前文件，也不在当前组装上绘制。':'');$('legend').textContent=descriptions[view]+versionWarning+(blinkTimer!==null?' 当前每 500ms 切换；点击停止。':'');for(const button of $('modes').querySelectorAll('[data-mode]'))button.classList.toggle('active',button.dataset.mode===view)}
$('region').addEventListener('change',()=>{stopBlink();region=DATA.regions[Number($('region').value)];draw()});$('zoom').addEventListener('change',draw);$('markers').addEventListener('change',draw);$('markerLabels').addEventListener('change',draw);
for(const button of $('modes').querySelectorAll('[data-mode]'))button.addEventListener('click',()=>{stopBlink();mode=button.dataset.mode;draw()});
$('opacity').addEventListener('input',()=>{stopBlink();mode='overlay';$('opacityValue').textContent=$('opacity').value+'%';draw()});
$('blink').addEventListener('click',()=>{if(blinkTimer!==null){stopBlink();draw();return}blinkMode='source';blinkTimer=setInterval(()=>{blinkMode=blinkMode==='source'?'candidate':'source';draw()},500);$('blink').textContent='停止闪烁';draw()});
document.addEventListener('visibilitychange',()=>{if(document.hidden){stopBlink();draw()}});
$('canvas').addEventListener('mousemove',event=>{const rect=$('canvas').getBoundingClientRect();let localX=(event.clientX-rect.left)/rect.width*$('canvas').width;const localY=(event.clientY-rect.top)/rect.height*$('canvas').height;const view=blinkTimer===null?mode:blinkMode;let pane='';if(view==='side_by_side'){pane=localX>=region.width?'当前侧':'原图侧';localX%=region.width}const x=Math.max(region.box[0],Math.min(region.box[2]-1,region.box[0]+localX-0.5));const y=Math.max(region.box[1],Math.min(region.box[3]-1,region.box[1]+localY-0.5));$('sample').textContent=`${pane} 原画布像素中心坐标 x=${x.toFixed(3)}, y=${y.toFixed(3)}。只读定位提示；不是已记录的观测证据。`});
draw().catch(error=>{$('legend').textContent='内嵌图像加载失败：'+error.message});
</script></body></html>'''


def create_review(source_path, candidate_path, settings, root, outdir):
    """Return root-relative report/html paths; all outputs must be new and under root."""
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("root must be a directory")
    if not isinstance(settings, dict):
        raise ValueError("settings must be an object")
    if not Path(source_path).is_absolute() or not Path(candidate_path).is_absolute():
        raise ValueError("source_path and candidate_path must be absolute Paths")
    source_path = inside(root, source_path, "source")
    candidate_path = inside(root, candidate_path, "candidate")
    output = inside(root, outdir, "output", must_exist=False)
    if output.exists():
        raise ValueError("output already exists; choose a new review directory")
    source_hash, candidate_hash = digest(source_path), digest(candidate_path)
    source, source_transparent = read_composite(source_path, "source")
    candidate, candidate_transparent = read_composite(candidate_path, "candidate")
    if source.size != candidate.size:
        raise ValueError("candidate canvas differs from source; no automatic registration is performed")
    regions = validate_regions(settings, source.size)
    limit = settings.get("max_error_px")
    if "max_error_px" in settings and (not finite_number(limit) or limit < 0):
        raise ValueError("max_error_px must be a finite nonnegative number; omit it for pending_limit")
    reference, observed = None, None
    reference_fresh, observed_fresh = False, False
    metadata = {}
    input_paths = [source_path, candidate_path]
    for key, kind, actual_hash in (("reference_points", "reference", source_hash),
                                   ("observed_points", "observed", candidate_hash)):
        if key not in settings:
            metadata[kind] = {"status": "unavailable", "reason": "Point file was not configured"}
            continue
        path = inside(root, settings[key], key, relative_only=True)
        input_paths.append(path)
        file_hash = digest(path)
        declared_hash, points = validate_point_file(path, kind, source.size)
        fresh = declared_hash == actual_hash
        metadata[kind] = {"path": relative(root, path), "sha256": file_hash,
                          "declared_image_sha256": declared_hash,
                          "status": "current" if fresh else "stale",
                          "point_count": len(points)}
        if kind == "reference":
            reference, reference_fresh = points, fresh
        else:
            observed, observed_fresh = points, fresh
    if reference is not None and observed is not None and reference_fresh and observed_fresh:
        unknown = sorted(set(observed) - set(reference))
        if unknown:
            raise ValueError("observed_points: unknown reference point IDs: " + ", ".join(unknown[:8]))
    pairs = validate_pairs(settings, reference)
    if any(path == output or path.is_relative_to(output) for path in input_paths):
        raise ValueError("output cannot overlap an input")
    # Detect changed inputs rather than binding a receipt to bytes other than those read.
    if digest(source_path) != source_hash or digest(candidate_path) != candidate_hash:
        raise ValueError("An input image changed during review; rerun on a stable file")
    for kind in ("reference", "observed"):
        if "path" in metadata[kind] and digest(root / metadata[kind]["path"]) != metadata[kind]["sha256"]:
            raise ValueError("A point file changed during review; rerun on a stable file")
    geometry = measure_geometry(reference, observed, reference_fresh, observed_fresh, limit)
    configuration = {"source": relative(root, source_path), "candidate": relative(root, candidate_path),
                     "output": relative(root, output), "settings": settings}
    hashes = {"configuration_canonical_sha256": canonical_digest(configuration),
              "source": {"path": relative(root, source_path), "sha256": source_hash},
              "candidate": {"path": relative(root, candidate_path), "sha256": candidate_hash},
              "reference": metadata["reference"], "observed": metadata["observed"]}
    fingerprint = canonical_digest({"schema": "alignment-review/v1", "hashes": hashes})
    report = {"schema": "alignment-review/v1", "status": geometry["status"], "visual_pass": None,
              "canvas": list(source.size), "configuration": configuration, "hashes": hashes,
              "fingerprint": fingerprint, "reference_fingerprint": metadata["reference"].get("sha256"),
              "geometry": geometry,
              "pairs": [pair_diagnostic(pair, reference, observed, reference_fresh and observed_fresh) for pair in pairs],
              "display": {"neutral_rgb": list(NEUTRAL_RGB), "source_had_transparency": source_transparent,
                          "candidate_had_transparency": candidate_transparent,
                          "candidate_transparency_filled_from_source": False,
                          "coordinate_convention": "Original decoded pixel-center indices: top-left pixel center is (0,0), x right, y down; ROI boxes right/bottom exclusive; canvas marks add 0.5 before display scaling"},
              "limitations": ["Numerical agreement is not visual acceptance; inspect the same-canvas overlays.",
                              "Fit points never vote. Check roles declare independence; this tool cannot verify their acquisition history.",
                              "Different parts can share source anchors, but shared anchors cannot replace independent checks elsewhere on each part.",
                              "Changing source points changes the reference and receipt fingerprint; old visual reviews cannot be reused.",
                              "Reference points are read-only. The tool never moves them or fits a candidate transform to improve errors.",
                              "Decimal coordinates express calculation precision. Manual observations around 0.5 px do not prove subpixel-perfect artwork.",
                              "Red/cyan gradients and RGB differences are appearance diagnostics, not contour-distance estimates."],
              "regions": []}
    output.mkdir(parents=True, exist_ok=False)
    viewer_regions = []
    for index, region in enumerate(regions):
        region_report, viewer = make_roi(region, index, source, candidate, root, output)
        report["regions"].append(region_report)
        viewer_regions.append(viewer)
    report_path, html_path = output / "alignment-report.json", output / "alignment-review.html"
    report["html"] = relative(root, html_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    viewer_data = {key: report[key] for key in ("status", "canvas", "hashes", "fingerprint", "geometry", "pairs")}
    viewer_data["regions"] = viewer_regions
    encoded = json.dumps(viewer_data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html_path.write_text(HTML_TEMPLATE.replace("__DATA__", encoded), encoding="utf-8")
    return {"status": report["status"], "report": relative(root, report_path),
            "html": relative(root, html_path), "fingerprint": fingerprint,
            "reference_fingerprint": report["reference_fingerprint"], "visual_pass": None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args(argv)
    try:
        path = Path(args.config).resolve(strict=True)
        config = read_json(path)
        root = path.parent
        source = inside(root, config.get("source"), "source", relative_only=True)
        candidate = inside(root, config.get("candidate"), "candidate", relative_only=True)
        output = inside(root, config.get("output"), "output", relative_only=True, must_exist=False)
        settings = {key: value for key, value in config.items() if key not in ("source", "candidate", "output")}
        result = create_review(source, candidate, settings, root, output)
    except (ValueError, OSError, Image.DecompressionBombError) as error:
        print(json.dumps({"status": "error", "message": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
