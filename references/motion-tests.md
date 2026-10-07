# 相对动作与状态诊断

拆件检查应让相邻部件相对运动，暴露隐藏底片与接头；联合动作再检查父子继承。幅度、状态和采样点来自本次动画要求，明确包含端点、中间姿态与恢复，不依赖正弦采样恰好碰到极值。

`motion_check.py` 仅变换已有素材，生成检查预览。它不生成缺失形体、眼开口蒙版或 Live2D／Spine 绑定；没有实际遮挡或变形素材时，不能据此宣称相关功能已实现。

## 输入

```text
python scripts/motion_check.py --parts project/output --manifest project/output/full-canvas-manifest.json --rig project/rig.json --project project/project.json --cases project/cases.json --cases-only --out project/qa/motion
```

`--parts` 是 manifest 中文件路径的解析根目录。rig 包含与原图一致的 `canvas`、`nodes`、`part_nodes` 和从后到前的 `order`。层序包含全部绘制段及隐藏状态，须与 project 一致；每个绘制段映射到控制节点，固定件显式映射为 `null`。父链不可成环。

| 节点类型 | 所需参数 |
| --- | --- |
| `rotate` | `root:[x,y]`，可选有符号 `gain`。 |
| `bend` | `root`、`tip`，可选 `lock_fraction`、`power`、`gain`。 |
| `scale` | `root`、`amplitude_fraction:[x,y]`。 |
| `translate` | `amplitude_px:[dx,dy]`。 |

节点可用 `parent` 指定继承。每个姿态的 `controls` 可覆盖节点的 `degrees` 或 `drive`；平移节点另支持实际像素 `translation_px`。大幅弯曲须检查逆变形残差和结果，诊断工具的变换不替代目标软件的变形器。

## 用例

`cases.json` 顶层为 `cases` 列表。每项有 `id`、`relation_ids`、`state_id`、`source_roi` 和 `poses`。用例 ID 应在项目 `test_cases` 及相关关系中登记；区域包含交界和整个约定运动范围。

`poses` 每项有唯一 `id`，可设置：

- `controls`：节点到实际控制量的映射。
- `state_id`：切换已存在的状态。
- `hide`：暂时隐藏绘制 ID 以查看底片，不改变正式交付。

底色增加独立显示、移动与恢复检查；投影检查投射者和接收面的相对变化。表情替换逐态检查，连续形变检查真实中间形体；不把两张替换图算成连续眨眼。

## 结果

每姿态保存全画布图、原尺寸 ROI、可见层、实际控制量、哈希及数值残差。联系图方便浏览，完整图用于找接口缺陷。`visual_status:pending` 留给实际检查，不因渲染成功自动变为通过。

按[桌面验收](quality.md)观察必测姿态，任一必查姿态失败则对应动作未通过。不要缩小幅度、裁去偏离内容或增加正常帧掩盖问题。素材、状态、rig 或动作幅度改变后，更新依赖并复查相关项。
