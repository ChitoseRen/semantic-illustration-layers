# 原图保留与隐藏区域重建

用于能保留原图可见像素的区域；确实无法复用的部件可完整重绘。纯固有色需要去除光照时，按[材质分层](material-layers.md)重建，不把原图阴影锁入底色。

## 生产蒙版

先核对目标及相邻件，区分外轮廓、遮挡边、结构线、材质边与投影边。规划框仅用于定位。每件准备同原图画布的：

| 字段 | 含义 |
| --- | --- |
| `source_layer` | 原图中本件实际可见内容的 RGBA，固定源坐标。 |
| `keep` | 可信、不透明、归属明确的保留区；RGB 保持原图，alpha 为 255。 |
| `unknown` | 待处理的混色边或归属不确定区域，不能进入 keep。 |
| `edit` | 允许补画、清理与衔接的区域，可与 unknown 重叠，不与 keep 重叠。 |

三个区域蒙版为 L 模式二值 0/255。`edit` 是编辑许可范围，不是最终轮廓。抗锯齿中混入邻件颜色时，先恢复前景颜色并检查黑白底，避免重复叠 alpha。

## 补画与取用

图像工具的编辑参考包含原图目标、接头和足够上下文，整图及已接受邻件提供画风与尺度依据。要求完整根部、末端、隐藏表面和动作余量，保留真实孔洞、细丝及有意出画的边界。生成画框不能截断目标；多张窄补丁接不上本体时，应扩大参考范围或重新制作完整候选。

先看候选完整轮廓，再局部试拼。清除邻件残影、旧遮挡线与不属于目标的内容；不要增加无依据的纹样或结构。工具没有像素蒙版接口时，以参考和文字表达意图，最终用本地蒙版约束采用范围，不声称生成器执行了硬蒙版。

候选按[对位规则](alignment-review.md)回到源画布后，准备：

- `target_alpha`：最终完整对象透明度，可见外轮廓参照原图，隐藏部分来自实际补画及连接需求。
- `generated_weight`：0 取 source_layer，255 取候选，中间值为内部颜色过渡；非零范围须在 edit 内，keep 内为零。
- `blend_allowed`：允许中间权重的二值区域；这里的源层、候选和最终 alpha 都须完全不透明。

混合只用于无结构线的同材质内部。轮廓和纹样不能靠交叉渐变对齐；像素来源须有足够 alpha 与实际前景颜色，不能使用透明像素下的 RGB 填洞。覆盖片的柔边是另外的 alpha 处理，先保证下方底片完整，不能与内部颜色权重混为一谈。

## 配置与前检

以下为字段示例。所有输入相对于配置文件，位于项目目录内；画布和区域按当前原图填写，先明确转换至 sRGB，避免误读嵌入 ICC。ID 使用字母、数字、下划线或连字符，保持唯一。

```json
{
  "source": "source.png",
  "output": "reconstruction",
  "critical_interface_ids": ["part_edge"],
  "interfaces": [{
    "id": "part_edge", "parts": ["part"], "kind": "outer",
    "roi": [0, 0, 64, 64], "front": "part", "line_owner": "part",
    "observed": true, "observation": "填写实际查看的原图和相邻件依据",
    "hidden_requirement": "填写需要延续的结构"
  }],
  "parts": [{
    "id": "part", "order": 10,
    "source_layer": "inputs/part-visible.png",
    "keep": "masks/keep.png", "unknown": "masks/unknown.png", "edit": "masks/edit.png",
    "candidate": "candidates/part.png",
    "registration": {"frame": [0, 0, 64, 64]},
    "target_alpha": "masks/complete-alpha.png",
    "generated_weight": "masks/generated-weight.png",
    "blend_allowed": "masks/blend-allowed.png"
  }]
}
```

这里的 64×64 是格式示例，不是角色预设。`interfaces.kind` 可为 `outer/occlusion/internal/material/shadow`；外轮廓与遮挡需有 `front`，`line_owner` 为相关部件或 `none`。纯遮挡的 `occlusion` 还需不同的 `front`、`back`。不明确的关键交界先查清；其他独立局部可继续。

```text
python scripts/mask_preflight.py reconstruction.json
python scripts/reconstruct_parts.py reconstruction.json
```

首次前检不需要候选、最终 alpha 和颜色权重。前检输出源像素回拼、单件预览、冲突及 `fingerprint`。实际检查后写入：

```json
{
  "preflight_review": {
    "status": "reviewed",
    "fingerprint": "本次前检返回的指纹",
    "evidence": "实际观察的位置、结论和证据"
  }
}
```

该记录是制作检查，不是新的用户审批。改变原图、选区或关系后重新前检，重建工具会拒绝过期记录。

每次重建输出到独立 `output/runs/<id>/`：`baseline_parts/` 为未校色版本，`parts/` 为校色候选，另有 before／after 透明复合与报告。没有校色配置时二者可能相同。查看并选择后才把对应文件写入生产项目，使用 `coordinate_space:"source_canvas"`，不再套生成候选的变换。输入不得被输出覆盖。

## 可选颜色补偿

几何和结构正确后，才对同一连续材质表面的慢变色偏作补偿。使用 `appearance_groups`，每组含 `id`、表面 `region`、`targets`、`samples:[{part,mask}]`、`sigma_px`、`min_support`、`max_correction`。参数来自实际取样，排除描线、其他材质、真实投影边和不确定区域；每个候选使用自己的对应样本。可用各部件 `color_protect` 保护不可改色区域。

脚本在线性 sRGB 中平滑加权颜色残差，只修允许范围内采用的生成像素，保持 keep 与最终 alpha。无可靠样本、支持不足或补偿过大时不外推。颜色补偿不能修正错位的阴影、纹样或形体，也不是通用重打光。

`appearance_check.py` 比较同背景的 `source`、`before`、`after` 不透明图。配置 `output` 为新目录，必填 `regions:[{id,box}]`；可选 `keep`、成对 `alpha_before/alpha_after`、`scales_px`，以及跨接缝 `profiles:[{id,points,expected}]`。expected 为 `continuous` 或 `source_discontinuity`，分别表示连续表面和原图本有的变化。

```text
python scripts/appearance_check.py appearance-review.json
```

查看原尺寸彩色、灰度、低频差异和接缝曲线，并复查单层边缘。校色候选必须与未校色版本比较，改善才采用；keep 的零差异只证明保留机制，隐藏区域也没有原图像素真值。最终结论按[统一验收](quality.md)记录。
