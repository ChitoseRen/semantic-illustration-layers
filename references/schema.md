# 项目数据与脚本接口

每张插画使用独立项目目录。配置里的路径相对于配置文件所在目录；随项目迁移的输入、素材和输出放在该目录内。画布尺寸从原图读取，坐标使用原图像素：左上角为原点，x 向右、y 向下，矩形为右下不包含的 `[left, top, right, bottom]`。

## 方案：plan.json

方案中的 `parts` 是逻辑部件；生产项目中的 `parts` 是实际绘制层／绘制段。使用 `drawings` 建立二者对应，不用绘制层数代替部件数。

| 字段 | 内容 |
| --- | --- |
| `source`、`purpose`、`motion_scope` | 原图路径、目标用途和动作范围。 |
| `parts[]` | 非空列表，每件有唯一 `id`、`name`、`purpose`、`regions`；区域用 `{"box":[x0,y0,x1,y1]}` 定位。 |
| `parts[].drawings` | 本部件拥有的绘制 ID；省略时使用部件 ID。 |
| `semantic_parent`、`motion_parent`、`depends_on` | 语义包含、运动继承和制作依赖，分别记录；方案中引用逻辑部件 ID。 |
| `drawing_specs` | 绘制 ID 到层序、可见性和[材质字段](material-layers.md)的映射。 |
| `materials` | 材质、本色判断依据及实际底色层；新项目必须填写。 |
| `relations`、`critical_relation_ids` | 遮挡记录及需要检查的关键交界清单。 |
| `states`、`test_cases`、`exclusive_groups` | 状态、诊断用例及不能同时显示的绘制层。 |
| `questions` | 会影响制作且尚未解决的问题。 |
| `confirmation` | 真实授权记录，见下文。 |

```text
python scripts/part_plan.py plan.json --out plan-preview
python scripts/part_plan.py plan.json --out plan-preview --check-confirmation
```

首条命令写出 HTML 与 `proposal.json`；展示 HTML 后按已有授权处理确认。`confirmation` 保存 `status: "confirmed"` 或 `"waived"`、`plan_sha256`、`source_sha256` 及非空 `user_message`。`waived` 仅用于用户确实授权省略确认的情况。校验缺失或过期记录时退出码为 2。

方案哈希不包含确认记录，并以原图内容哈希替代路径，因此仅搬迁项目不影响绑定。原图或方案改变时更新预览和绑定，是否需要再问由实际授权范围决定。生产素材和检查过程另存，不反复改动已确认方案。

## 制作：project.json

`source` 为必填原图；`output` 为项目内独立输出目录。`background` 可指向同画布的独立背景，不提供时保持透明，不以原图代替背景。`materials`、关系和状态由方案带入。

| parts[] 字段 | 内容 |
| --- | --- |
| `id`、`logical_id`、`name` | 唯一绘制 ID、所属逻辑部件及显示名称；ID 使用字母、数字、下划线或连字符。 |
| `asset` | 已有 RGBA 或带显式透明度的 PNG；缺失素材留缺失状态，不用原图裁切冒充补全件。 |
| `order`、`visible` | 从后到前的数值层序、默认可见性；同序时保持数组顺序。隐藏状态也需要真实素材。 |
| `crop` | 仅用于准备参考的原图矩形，不是生产蒙版。 |
| `coordinate_space` | `source_canvas` 表示已在原画布定位，要求原尺寸和单位变换；否则使用 `asset` 或省略。 |
| `registration` | 原始候选到画布的 `frame` 或 `landmarks`，见[对位说明](alignment-review.md)。 |
| `material_id`、`paint_role`、`receiver_id`、`caster_ids`、`caster_note`、`motion_policy` | 每个绘制层的材质、绘画职责及光影关系。 |
| `semantic_parent`、`motion_parent` | 引用逻辑 ID，同一逻辑部件的各绘制段须一致，父链不可成环。 |
| `depends_on` | 引用制作所依赖的绘制 ID，不决定层序。 |

下面是单个绘制层的字段示例；完整项目还须提供材质、其他层和实际关系：

```json
{
  "id": "part_base",
  "logical_id": "part",
  "name": "目标部件／固有色",
  "asset": "assets/part-base.png",
  "coordinate_space": "source_canvas",
  "order": 10,
  "visible": true,
  "material_id": "surface",
  "paint_role": "base_color"
}
```

## 遮挡、状态与动作

关系端点引用**绘制 ID**。同一对象在不同区域前后穿插时，使用共享 `logical_id` 的绘制段，逐区域检查层序，不强行把逻辑对象的所有遮挡排成一条全局链。

| relations[] 字段 | 内容 |
| --- | --- |
| `id`、`front`、`back` | 稳定关系 ID，前后两个不同的绘制端点。 |
| `source_roi` | 原图内包含交界及运动范围的检查矩形。 |
| `line_owner` | 拥有交界线的端点；尚不确定时为 `unknown`。 |
| `connection_points` | `{id, point:[x,y], drawing_id?}` 列表。只有遮挡、没有实际连接时填 `[]`，并以 `connection_points_absence` 解释。 |
| `hidden_requirements` | 需要完整承接的区域与结构；无隐藏延伸需求时说明依据。 |
| `states`、`test_cases` | 本关系适用的状态和用例 ID。 |
| `observation` | `{status:"observed"或"unknown", evidence:"实际观察依据"}`。 |

`critical_relation_ids` 保存本次盘点的关键关系，不因缺记录而删掉。缺项保持待查，不自动算作全覆盖。

`states` 为 `{id, visible:[绘制ID]}` 列表，每个状态给出**完整可见集合**；未声明 `default` 时按各件 `visible` 决定。`exclusive_groups` 中每组至多显示一个绘制层。`test_cases` 用 `{id,state_id}` 登记用例，具体姿态另存，见[动作诊断](motion-tests.md)。

`relations.py` 检查引用、父链、制作依赖、材质及各状态的有效层序。它不判断对象是否画对。未提供关系或观察依据时，相关覆盖保持未知。

## 组装命令

```text
python scripts/asset_pipeline.py prepare project/project.json
python scripts/asset_pipeline.py register project/project.json
python scripts/asset_pipeline.py compose project/project.json
python scripts/asset_pipeline.py compose project/project.json --state expression
```

- `prepare` 输出参考裁切与 `prepared.json`。
- `register` 将已有素材定位为 `parts/<id>.png`，在 `registration.json` 中记录变换、哈希及 `ready/missing/invalid`。
- `compose` 重新读取当前素材并注册，输出透明 `foreground.png`、有独立背景时的 `composite.png`、`full-canvas-manifest.json` 和 `composition.json`。命名状态带状态后缀。
- 素材不全时预览带 `.incomplete`，当前清单中的缺失文件为 `null`。只依当前报告取用结果，不扫描旧输出猜测当前文件。

退出码 0 表示本次几何操作完成，2 表示素材不全但已写出诊断，1 表示输入或操作错误。这些状态不表示视觉验收完成。

## 检查记录与复用

`fidelity_report.py` 可比较同画布、不透明的原图与候选复合，需要原图主体蒙版；候选主体 alpha 可另外提供。带 `--manifest` 时会建立六项部件检查 CSV。

```text
python scripts/fidelity_report.py --source project/source.png --candidate project/output/composite.png --source-mask project/masks/subject.png --candidate-alpha project/output/foreground.png --manifest project/output/full-canvas-manifest.json --out project/qa/fidelity
```

同一 `logical_id` 的绘制段合为一条部件记录。多绘制段的 `evidence` 和 `asset_sha256` 分别用 JSON 对象覆盖每个绘制 ID；单段可以使用非空证据文字与一个哈希。证据不足、文件缺失或哈希改变时保持待查。素材路径由当前 manifest 的 `file` 解析，不以评审 CSV 指向的其他文件替代。

`dependency_fingerprint(project, drawing_ids, asset_hashes, rig=None, relation_ids=None, state_ids=None, evidence_kind="motion")` 生成检查依赖指纹。传入全部当前注册素材哈希及 `__source__` 原图哈希；`motion` 包含关联资产、状态、层序与运动祖先，`static` 不含 rig，`alpha` 仅绑定本件及原图。`affected_drawings(before, after, changed_ids)` 提供变更前后相关部件与后代的并集。采用这些辅助函数时，保存当次依赖清单，按[验收规则](quality.md)决定哪些证据可以复用。
