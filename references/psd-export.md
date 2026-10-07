# 环境与双 PSD 导出

## 环境准备

确认可用的图像编辑／生成工具、桌面控制器、图层编辑软件，以及 Python 3.10+、Pillow、NumPy。路径从当前技能和项目目录发现，不写死账户、插件版本或旧文档 ID。只安装实际缺失的依赖；Windows 中文路径可用 `python -X utf8`。

随包组装器使用 Photoshop ExtendScript。读取当前连接的实际工具签名；先用独立测试文档验证同画布 PNG 导入、透明度、保存及重开，不影响已有原稿。桌面控制按对应技能检查连接，失败处理见[验收说明](quality.md)。

## 两份文件的职责

| 文件 | 保留内容 |
| --- | --- |
| 可编辑母版 | 独立固有色、线稿、纹理、阴影、高光，以及实际使用的蒙版、剪贴和混合关系。 |
| 普通图层版 | 同一动画绘制段的绘画效果合成到普通透明像素层；独立运动部件、前后段、隐藏底片及替换状态仍分开。 |

两份均用唯一、稳定的部件名称，RGB、8 bit/channel、同一画布和色彩解释，分别保存到新文件。原图有背景则独立保存背景；透明原图不新增场景。原图参考在交付中默认隐藏，不能参与候选复合或补洞。

母版的固有色与光影必须真正可编辑。内置脚本可以逐层导入普通 RGBA；需要组、剪贴、特殊混合或蒙版时，在 Photoshop 中实际建立。普通版从已确认母版按绘制段合成，不跨越需要相对运动的对象，不把整图合为一层。两份文件分别命名、分别检查。

## 组装器接口

`scripts/build_psd.jsx` 从 `SEMANTIC_PSD_REQUEST` 读取参数，返回 JSON 字符串。直接执行 raw ExtendScript 时可使用：

```javascript
(function () {
  var SEMANTIC_PSD_REQUEST = {
    phase: 'begin', name: 'Animation layers', canvas: [width, height]
  };
  return $.evalFile(new File(skillDirectory + '/scripts/build_psd.jsx'));
})();
```

先在当前调用中定义实际 `width`、`height` 和 `skillDirectory`。若连接明确把输入包在函数内，按其返回约定调用；不要假定 ExtendScript 自带 `JSON`。

| phase | 参数及作用 |
| --- | --- |
| `begin` | `name`、`canvas:[宽,高]`，可选 `resolution`；新建 RGB/8 文档并返回 ID。 |
| `add` | 真实 `document_id`，1–12 项 `layers:[{file:PNG绝对路径,name,visible}]`；从后到前导入。 |
| `inspect` | `document_id`；读取实际图层。部分导入失败后先检查，再补剩余项目。 |
| `finish` | `document_id`、不同于原稿的新 `psd` 路径和 `preview` PNG 路径，父目录须已存在。 |

所有 PNG 保留完整画布和透明余量，不裁紧后让软件自动居中。组装器不会创建剪贴、特殊混合、状态或动画绑定，也不关闭目标文档。写入前后检查响应，不因文件存在就认为导入完整。

原图参考层最后导入并设为隐藏；设置活动层、锁定或图层复合后再核对其实际可见性。表情用图层复合保存时，确认同时记录需要的可见性和位置；`layerComps.add(name, comment, appearance, position, visibility)` 的最后一个参数是可见性。

## 保存与重开

两份文件逐一保存、重开，并按[统一验收](quality.md)操作真实图层：母版检查材质层独立性和混合关系，普通版检查像素部件、底片和状态。对照两份同状态复合与原图，发现差异先区分素材、层序、导入位置及色彩管理。恢复默认状态、隐藏参考，再核对最终保存文件。

## Live2D 与 Spine

普通图层版作为导入起点。版本或目标不同会影响支持的混合与蒙版，实际导入前核对当前官方说明：

- [Live2D：PSD 制作注意事项](https://docs.live2d.com/en/cubism-editor-manual/precautions-for-psd-data/)要求 RGB、8 bit/channel，并说明部件合并与蒙版处理；[导入说明](https://docs.live2d.com/en/cubism-editor-manual/psd-import/)说明隐藏层也会导入。隐藏参考层不能因此当作已排除，导入后应明确移除或排除它，保留需要的表情附件。
- [Spine：Import PSD](https://esotericsoftware.com/spine-import-psd)说明像素层、层序、隐藏层选项和 `[ignore]` 标签。可用 `reference_original [ignore]` 排除参考；若还要导入隐藏表情，不应无差别忽略所有隐藏层。

普通像素层版本是本技能的交付约定，不代表所有版本都只支持普通混合。实际导入时检查层数、命名、位置、透明度和状态；完成后再记录使用的软件版本与结果。未执行导入、绑定或动画制作时，如实列为未验证／未制作。
