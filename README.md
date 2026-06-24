# SAM3 DETR Modular and LoRA

这个仓库当前主要整理的是 [`sam3_detr_exp`](/slow_disk/ccl/codes/sam3/sam3_detr_exp) 这条实验主线，目标很直接：

- 把原始 `sam3.pt` 拆成清楚的模块
- 保持非 JIT、可继续训练、可单独替换模块
- 验证模块化推理结果和原始 SAM3 一致
- 在 modular DETR 上做 LoRA 微调，并直观看到前后效果变化

如果你是第一次看这个项目，先从这里开始就够了。

## Project Layout

### [`sam3_detr_exp/run_video_det_modular.py`](/slow_disk/ccl/codes/sam3/sam3_detr_exp/run_video_det_modular.py)

从原始 `sam3.pt` 导出模块化权重，生成 `sam3_detr_exp/weights_modular/*.pt`。

```bash
source /slow_disk/ccl/codes/sam3/.venv/bin/activate
python sam3_detr_exp/run_video_det_modular.py
```

### [`sam3_detr_exp/modular_pipeline.py`](/slow_disk/ccl/codes/sam3/sam3_detr_exp/modular_pipeline.py)

模块化组装核心。负责把各个 `*.pt` 权重重新拼成 detector、tracker 和 video predictor。

常用入口：

- `build_detector_modules()`
- `build_detector_model()`
- `build_tracker_modules()`
- `build_tracker_model()`
- `build_video_model()`
- `ModularVideoPredictor`

### [`sam3_detr_exp/run_detr_prompt_inference.py`](/slow_disk/ccl/codes/sam3/sam3_detr_exp/run_detr_prompt_inference.py)

只跑 detector 提示推理，支持文本提示和框提示，也支持加载 LoRA。

文本提示：

```bash
python sam3_detr_exp/run_detr_prompt_inference.py \
  --image assets/images/test_image.jpg \
  --text shoe \
  --output sam3_detr_exp/outputs/detr_text_prompt.png
```

加载 LoRA：

```bash
python sam3_detr_exp/run_detr_prompt_inference.py \
  --image assets/images/test_image.jpg \
  --text "linear crack" \
  --lora sam3_detr_exp/weights_lora/detr_transformer_lora.pt \
  --output sam3_detr_exp/outputs/detr_text_prompt_lora.png
```

### [`sam3_detr_exp/compare_image_original_vs_modular.py`](/slow_disk/ccl/codes/sam3/sam3_detr_exp/compare_image_original_vs_modular.py)

对比原始 `sam3.pt` 和模块化 detector 在同一张图上的结果。

```bash
python sam3_detr_exp/compare_image_original_vs_modular.py \
  --image assets/images/test_image.jpg \
  --prompt shoe
```

### [`sam3_detr_exp/compare_video_original_vs_modular.py`](/slow_disk/ccl/codes/sam3/sam3_detr_exp/compare_video_original_vs_modular.py)

对比原始 `sam3.pt` 和模块化 video pipeline 在同一段视频上的结果。

```bash
python sam3_detr_exp/compare_video_original_vs_modular.py \
  --video assets/videos/bedroom.mp4 \
  --prompt person \
  --max-frames 2
```

### [`sam3_detr_exp/train_detr_lora.py`](/slow_disk/ccl/codes/sam3/sam3_detr_exp/train_detr_lora.py)

基于 `lightning==2.6.5` 的 DETR LoRA 微调入口，当前接的是 `/slow_disk/ccl/data/crack_segment` 的 YOLO segmentation 数据。

最小 dry-run：

```bash
python sam3_detr_exp/train_detr_lora.py \
  --dataset-root /slow_disk/ccl/data/crack_segment \
  --train-split train \
  --val-split val \
  --max-train-samples 1 \
  --max-val-samples 1 \
  --dry-run
```

正式训练：

```bash
python sam3_detr_exp/train_detr_lora.py \
  --dataset-root /slow_disk/ccl/data/crack_segment \
  --train-split train \
  --val-split val \
  --batch-size 20 \
  --epochs 20
```

默认行为：

- 输入是图像加文本提示
- 监督输出是 `pred_logits`、`pred_boxes`、`pred_masks`
- 文本提示默认来自 `data.yaml` 的类别名
- LoRA 权重默认保存到 `sam3_detr_exp/weights_lora/detr_transformer_lora.pt`

## LoRA Effect

下面这两张图是同一套 detector 推理脚本导出的结果，方便直接看 LoRA 微调前后差异。

### Before LoRA

![Before LoRA](sam3_detr_exp/outputs/detr_text_prompt.png)

### After LoRA

![After LoRA](sam3_detr_exp/outputs/detr_text_prompt_lora.png)

这两张图对应的推理命令分别是：

```bash
python sam3_detr_exp/run_detr_prompt_inference.py \
  --image assets/images/test_image.jpg \
  --text "linear crack" \
  --output sam3_detr_exp/outputs/detr_text_prompt.png
```

```bash
python sam3_detr_exp/run_detr_prompt_inference.py \
  --image assets/images/test_image.jpg \
  --text "linear crack" \
  --lora sam3_detr_exp/weights_lora/detr_transformer_lora.pt \
  --output sam3_detr_exp/outputs/detr_text_prompt_lora.png
```

## Training Structure

LoRA 训练代码已经从单文件脚本提炼成分层结构：

- [`sam3_detr_exp/model/detr_lora_module.py`](/slow_disk/ccl/codes/sam3/sam3_detr_exp/model/detr_lora_module.py)
  - LightningModule 封装
- [`sam3_detr_exp/utils/detr_lora_data.py`](/slow_disk/ccl/codes/sam3/sam3_detr_exp/utils/detr_lora_data.py)
  - YOLO segmentation dataset 和 datamodule
- [`sam3_detr_exp/utils/detr_lora_utils.py`](/slow_disk/ccl/codes/sam3/sam3_detr_exp/utils/detr_lora_utils.py)
  - LoRA 挂载、保存加载、target 构造、loss 和 detector 组装

## More Docs

- 模块输入输出、shape、数据流图：
  [`sam3_detr_exp/docs/modular-weights.md`](/slow_disk/ccl/codes/sam3/sam3_detr_exp/docs/modular-weights.md)
- DETR LoRA 微调方案：
  [`sam3_detr_exp/docs/detr-lora-finetune.md`](/slow_disk/ccl/codes/sam3/sam3_detr_exp/docs/detr-lora-finetune.md)

## Recommended Order

如果你想完整复现这条链路，按这个顺序跑最省事：

1. 导出模块权重
2. 跑 detector-only 推理
3. 对比原始模型和模块化模型
4. 训练 DETR LoRA
5. 用训练前后两张可视化图检查 LoRA 效果
