# 模块化权重与输入输出说明 (Modular Weights and I/O Guide)

本说明对应 `sam3_detr_exp/run_text_det_modular.py` 脚本，支持把 SAM3 拆成独立模块权重，并导出一次推理的输入输出维度记录，便于未来重新组合或复用。

## 1. 权重文件 (Weight Files)

运行脚本后会生成以下文件：

- `sam3_detr_exp/weights_modular/vision_backbone.pt`
- `sam3_detr_exp/weights_modular/text_encoder.pt`
- `sam3_detr_exp/weights_modular/vl_backbone.pt`
- `sam3_detr_exp/weights_modular/transformer.pt`
- `sam3_detr_exp/weights_modular/segmentation_head.pt`
- `sam3_detr_exp/weights_modular/geometry_encoder.pt`
- `sam3_detr_exp/weights_modular/dot_product_scoring.pt`
- `sam3_detr_exp/weights_modular/full_model.pt`

说明：
- 每个文件只包含对应模块的 `state_dict()`。
- 重新加载时必须使用相同的模块构造函数与配置。
- `full_model.pt` 是完整模型的 `state_dict()`，用于整体恢复。

## 2. 输入输出维度记录 (I/O Shape Record)

脚本会保存一次推理时的实际维度信息：

- `sam3_detr_exp/runs/text_det_modular_shapes.json`

该文件是一次运行的真实维度快照，包含：
- `image_input`: 模型输入图像张量的维度
- `backbone_out`: 视觉骨干输出（含多尺度特征）
- `text_outputs`: 文本编码输出
- `outputs`: detector 输出（`pred_boxes`/`pred_logits`/`pred_masks` 等）
- `postprocess`: 后处理后的 `boxes`/`masks`/`scores` 维度

> 注意：维度会随输入分辨率、batch size、文本数量和配置变化。

## 3. 模块输入输出语义 (Module I/O Semantics)

以下是模块的输入输出语义概览（维度请以 `text_det_modular_shapes.json` 为准）：

- 视觉骨干 (vision_backbone)
  - 输入: `image_t` (B, 3, H, W)
  - 输出: `backbone_fpn` (多尺度特征列表), `vision_pos_enc`, `sam2_backbone_out`

- 文本编码 (text_encoder)
  - 输入: 文本列表 (List[str])
  - 输出: `language_features`, `language_mask`

- 视觉-语言骨干 (vl_backbone)
  - 输入: 图像张量 + 文本列表
  - 输出: `backbone_out` 字典 (视觉 + 文本特征)

- Transformer (transformer)
  - 输入: 图像特征 + 文本特征 + 几何提示
  - 输出: 查询特征与中间预测

- 分割头 (segmentation_head)
  - 输入: transformer 解码特征
  - 输出: mask logits

- 几何编码 (geometry_encoder)
  - 输入: 几何提示 + 视觉特征
  - 输出: 几何嵌入

- 打分头 (dot_product_scoring)
  - 输入: 查询特征 + 文本特征
  - 输出: 概念匹配分数

## 4. 重新加载示例 (Reload Example)

```python
import torch
from sam3.model_builder import (
    _create_dot_product_scoring,
    _create_geometry_encoder,
    _create_sam3_transformer,
    _create_segmentation_head,
    _create_text_encoder,
    _create_vl_backbone,
    _create_vision_backbone,
)

vision_backbone = _create_vision_backbone(compile_mode=None, enable_inst_interactivity=True)
text_encoder = _create_text_encoder("sam3/assets/bpe_simple_vocab_16e6.txt.gz")
vl_backbone = _create_vl_backbone(vision_backbone, text_encoder)
transformer = _create_sam3_transformer()
segmentation_head = _create_segmentation_head()
geometry_encoder = _create_geometry_encoder()
dot_prod_scoring = _create_dot_product_scoring()

vision_backbone.load_state_dict(torch.load("sam3_detr_exp/weights_modular/vision_backbone.pt"))
text_encoder.load_state_dict(torch.load("sam3_detr_exp/weights_modular/text_encoder.pt"))
```
