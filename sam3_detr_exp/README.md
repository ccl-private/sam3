# sam3_detr_exp

这个目录现在只保留一条非 JIT 的模块化主线：

- 从原始 `sam3.pt` 导出模块权重
- 用 `weights_modular/*.pt` 重新组装 detector / tracker
- 对比原始模型和模块化模型的推理结果
- 单独跑 detector 提示推理

如果你只想知道“这里每个文件是干什么的、怎么用”，先看这份 README。
如果你要看模块输入输出、shape、数据流图，再看 [docs/modular-weights.md](/slow_disk/ccl/codes/sam3/sam3_detr_exp/docs/modular-weights.md)。

## Directory Overview

### [run_video_det_modular.py](/slow_disk/ccl/codes/sam3/sam3_detr_exp/run_video_det_modular.py)

用途：

- 从原始 `sam3.pt` 拆出模块权重
- 生成 `weights_modular/*.pt`

什么时候用：

- 第一次准备模块化权重时
- 原始 checkpoint 更新后，想重新导出模块权重时

怎么用：

```bash
source /slow_disk/ccl/codes/sam3/.venv/bin/activate
python sam3_detr_exp/run_video_det_modular.py
```

输出：

- `sam3_detr_exp/weights_modular/*.pt`

### [modular_pipeline.py](/slow_disk/ccl/codes/sam3/sam3_detr_exp/modular_pipeline.py)

用途：

- 这是整个目录的核心组装入口
- 负责从 `weights_modular/*.pt` 组装：
  - detector
  - tracker
  - video model

主要接口：

- `build_detector_modules()`
- `build_detector_model()`
- `build_tracker_modules()`
- `build_tracker_model()`
- `build_video_model()`
- `ModularVideoPredictor`

什么时候用：

- 你写自己的推理脚本时
- 你后面想做模块级微调 / 蒸馏 / ONNX 包装时
- 你想单独调用某个模块做 I/O 测试时

### [compare_image_original_vs_modular.py](/slow_disk/ccl/codes/sam3/sam3_detr_exp/compare_image_original_vs_modular.py)

用途：

- 在同一张图上对比：
  - 原始 `sam3.pt`
  - 模块化 `weights_modular`

什么时候用：

- 验证模块化 detector 的结果是否和原始模型一致
- 快速肉眼对比 box / mask 是否重合

怎么用：

```bash
python sam3_detr_exp/compare_image_original_vs_modular.py \
  --image assets/images/test_image.jpg \
  --prompt shoe
```

默认输出：

- `sam3_detr_exp/outputs/image_original_vs_modular.png`

### [compare_video_original_vs_modular.py](/slow_disk/ccl/codes/sam3/sam3_detr_exp/compare_video_original_vs_modular.py)

用途：

- 在同一段视频上对比：
  - 原始 `sam3.pt`
  - 模块化 `weights_modular`

什么时候用：

- 验证模块化 video pipeline 是否和原始模型一致
- 看 tracking / id / mask 是否明显分叉

怎么用：

```bash
python sam3_detr_exp/compare_video_original_vs_modular.py \
  --video assets/videos/bedroom.mp4 \
  --prompt person \
  --max-frames 2
```

默认输出：

- `sam3_detr_exp/outputs/video_original_vs_modular.mp4`
- `sam3_detr_exp/outputs/video_original_vs_modular.png`

说明：

- `png` 是首帧预览
- `mp4` 是逐帧对比视频

### [run_detr_prompt_inference.py](/slow_disk/ccl/codes/sam3/sam3_detr_exp/run_detr_prompt_inference.py)

用途：

- 只跑 modular detector
- 支持两种提示：
  - 文本提示 `--text`
  - 框提示 `--box`

什么时候用：

- 你只想验证 DETR 那半边
- 你想单独看 detector 的目标分割结果
- 你后面想把 detector 单独抽出来时

怎么用：

文本提示：

```bash
python sam3_detr_exp/run_detr_prompt_inference.py \
  --image assets/images/test_image.jpg \
  --text shoe \
  --output sam3_detr_exp/outputs/detr_text_prompt.png
```

框提示：

```bash
python sam3_detr_exp/run_detr_prompt_inference.py \
  --image assets/images/test_image.jpg \
  --box 320,300,980,690 \
  --output sam3_detr_exp/outputs/detr_box_prompt.png
```

说明：

- `--box` 格式是像素坐标 `x0,y0,x1,y1`
- 脚本内部会自动转成模型需要的归一化 `cxcywh`

### [weights_modular/](/slow_disk/ccl/codes/sam3/sam3_detr_exp/weights_modular)

用途：

- 保存模块化拆分后的 `state_dict`

里面包括：

- `vision_backbone.pt`
- `text_encoder.pt`
- `transformer_encoder.pt`
- `transformer_decoder.pt`
- `segmentation_head.pt`
- `geometry_encoder.pt`
- `dot_product_scoring.pt`
- `tracker_sam_heads.pt`
- `tracker_maskmem_backbone.pt`
- `tracker_transformer.pt`

说明：

- 这些不是可直接裸跑的计算图
- 它们必须通过 [modular_pipeline.py](/slow_disk/ccl/codes/sam3/sam3_detr_exp/modular_pipeline.py) 重新组装

### [docs/modular-weights.md](/slow_disk/ccl/codes/sam3/sam3_detr_exp/docs/modular-weights.md)

用途：

- 模块化权重和模块接口说明书
- 包含：
  - 每个模块输入输出
  - 实测 shape
  - detector / tracker 数据流图
  - 当前目录最终工作流

什么时候看：

- 想理解模块边界时
- 想做模块级微调 / 蒸馏 / ONNX 时
- 想确认每个模块实际吃什么、吐什么时

### [outputs/](/slow_disk/ccl/codes/sam3/sam3_detr_exp/outputs)

用途：

- 保存对比图、对比视频、detector 可视化结果

说明：

- 这是运行产物目录
- 已经在 `.gitignore` 里忽略，不会默认提交

## Recommended Usage

### 1. 先导出模块权重

```bash
source /slow_disk/ccl/codes/sam3/.venv/bin/activate
python sam3_detr_exp/run_video_det_modular.py
```

### 2. 验证 detector-only

```bash
python sam3_detr_exp/run_detr_prompt_inference.py \
  --image assets/images/test_image.jpg \
  --text shoe
```

### 3. 验证图片结果

```bash
python sam3_detr_exp/compare_image_original_vs_modular.py \
  --image assets/images/test_image.jpg \
  --prompt shoe
```

### 4. 验证视频结果

```bash
python sam3_detr_exp/compare_video_original_vs_modular.py \
  --video assets/videos/bedroom.mp4 \
  --prompt person \
  --max-frames 2
```

## Which File To Use

如果你现在的目标是：

- “我要重新导出模块权重”
  - 用 `run_video_det_modular.py`

- “我要写自己的 modular 推理代码”
  - 用 `modular_pipeline.py`

- “我要看原始模型和模块化模型是不是一样”
  - 用 `compare_image_original_vs_modular.py`
  - 或 `compare_video_original_vs_modular.py`

- “我只想测试 detector，从文本或框提示直接到分割结果”
  - 用 `run_detr_prompt_inference.py`

- “我想知道模块接口、shape、数据流”
  - 看 `docs/modular-weights.md`
