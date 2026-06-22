# SAM3 Modular Weights Guide

`sam3_detr_exp/weights_modular/*.pt` 保存的是各子模块的 `state_dict`，不是可直接裸执行的计算图。

这条链路的目标是：

- 把 `sam3.pt` 拆成多个可单独替换、单独训练、单独加载的模块
- 仍然保持和原始 `sam3.pt` 一样的推理结果
- 后续可以把最小推理 runtime 从主仓库里继续抽离出去

当前模块组装入口在 [modular_pipeline.py](/slow_disk/ccl/codes/sam3/sam3_detr_exp/modular_pipeline.py)。

## Final Layout

当前 `sam3_detr_exp/` 最终只保留这条非 JIT 主线需要的内容：

- `run_video_det_modular.py`
  - 从原始 `sam3.pt` 导出 `weights_modular/*.pt`

- `modular_pipeline.py`
  - 唯一的模块化组装入口
  - 负责从 `weights_modular/*.pt` 组装 detector / tracker / video model

- `compare_image_original_vs_modular.py`
  - 在单张图片上对比：
    - 原始 `sam3.pt`
    - 模块化 `weights_modular`

- `compare_video_original_vs_modular.py`
  - 在单段视频上对比：
    - 原始 `sam3.pt`
    - 模块化 `weights_modular`

- `weights_modular/`
  - 非 JIT 模块权重目录

- `docs/modular-weights.md`
  - 本说明文档

## Final Workflow

### 1. 导出模块权重

```bash
source /slow_disk/ccl/codes/sam3/.venv/bin/activate
python sam3_detr_exp/run_video_det_modular.py
```

### 2. 跑图片对比

```bash
python sam3_detr_exp/compare_image_original_vs_modular.py \
  --image assets/images/test_image.jpg \
  --prompt shoe
```

默认输出：

- `sam3_detr_exp/outputs/image_original_vs_modular.png`

左侧是原始 `sam3.pt`，右侧是模块化 `weights_modular`。

### 3. 跑视频对比

```bash
python sam3_detr_exp/compare_video_original_vs_modular.py \
  --video assets/videos/bedroom.mp4 \
  --prompt person
```

默认输出：

- `sam3_detr_exp/outputs/video_original_vs_modular.mp4`
- `sam3_detr_exp/outputs/video_original_vs_modular.png`

其中：

- `mp4` 是整段视频逐帧对比
- `png` 是首帧预览，方便快速确认左右是否一致

## How To Verify

你现在只需要看两件事：

1. 图片对比图
   - 左右目标数量是否一致
   - box 位置是否一致
   - mask 轮廓是否基本重合

2. 视频对比视频
   - 左右同一帧的目标 id / mask / box 是否基本一致
   - 遮挡、出入画、持续跟踪时是否明显分叉

## End-to-End Data Flow

下面这张图描述的是当前非 JIT `weights_modular` 主链的完整数据流。

```text
Image / Video Frame
        |
        v
+-------------------+
| preprocessing     |
| resize / normalize|
+-------------------+
        |
        v
+-------------------+          Text Prompt
| vision_backbone   |<-------------------+
| .pt               |                    |
+-------------------+                    |
        |                                |
        | backbone_fpn / pos_enc         |
        v                                |
+-------------------+                    |
| SAM3VLBackbone    |                    |
| forward_image     |                    |
+-------------------+                    |
        |                                |
        |                                v
        |                      +-------------------+
        |                      | text_encoder.pt   |
        |                      +-------------------+
        |                                |
        |                                | language_features
        |                                | language_mask
        |                                v
        +---------------------->+-------------------+
                                | SAM3VLBackbone    |
                                | forward_text      |
                                +-------------------+
                                           |
                                           | text tokens
                                           v
                         Geometric Prompt / dummy prompt
                                           |
                                           v
                                +-------------------+
                                | geometry_encoder  |
                                | .pt               |
                                +-------------------+
                                           |
                                           | geo prompt tokens
                                           v
                                +-------------------+
                                | _encode_prompt    |
                                | text + geo concat |
                                +-------------------+
                                           |
                                           | prompt / prompt_mask
                                           v
                                +-------------------+
                                | transformer_      |
                                | encoder.pt        |
                                +-------------------+
                                           |
                                           | memory / pos_embed
                                           | spatial_shapes
                                           v
                                +-------------------+
                                | transformer_      |
                                | decoder.pt        |
                                +-------------------+
                                           |
                              +------------+------------+
                              |                         |
                              v                         v
                    +-------------------+     +-------------------+
                    | dot_product_      |     | segmentation_head |
                    | scoring.pt        |     | .pt               |
                    +-------------------+     +-------------------+
                              |                         |
                              | pred_logits             | pred_masks
                              +------------+------------+
                                           |
                                           v
                                +-------------------+
                                | detector outputs  |
                                | boxes / scores /  |
                                | masks             |
                                +-------------------+
                                           |
                                           |
                 +-------------------------+--------------------------+
                 |                                                    |
                 | image mode                                         | video mode
                 |                                                    |
                 v                                                    v
       final image boxes/masks                          tracker initialization / update
                                                                  |
                                                                  v
                                                      +------------------------+
                                                      | tracker_sam_heads.pt   |
                                                      | SAM prompt + SAM mask  |
                                                      | decoder related params |
                                                      +------------------------+
                                                                  |
                                                                  | low/high-res masks
                                                                  | obj_ptr / obj_score
                                                                  v
                                                      +------------------------+
                                                      | tracker_maskmem_       |
                                                      | backbone.pt            |
                                                      +------------------------+
                                                                  |
                                                                  | mask memory features
                                                                  v
                                                      +------------------------+
                                                      | tracker_transformer.pt |
                                                      +------------------------+
                                                                  |
                                                                  | propagated memory
                                                                  v
                                                      +------------------------+
                                                      | tracker outputs        |
                                                      | per-frame masks / ids  |
                                                      | / scores               |
                                                      +------------------------+
                                                                  |
                                                                  v
                                                      final video outputs
```

如果只看主干，可以把它压缩成两段：

1. detector
   `image + text (+ geo prompt) -> boxes / scores / masks`

2. tracker
   `detector masks -> memory encoding -> temporal propagation -> video masks / ids / scores`

## Detector Flow

下面这张图只看单帧 detector。

```text
image
  |
  v
preprocess
  |
  v
vision_backbone.pt
  |
  |--> backbone_fpn:
  |    [B,256,288,288]
  |    [B,256,144,144]
  |    [B,256,72,72]
  |
  |--> vision_pos_enc:
  |    same 3 scales
  |
  `--> vision_features:
       [B,256,72,72]

text prompt
  |
  v
text_encoder.pt
  |
  |--> language_features: [Seq,B,256]
  |--> language_mask:     [B,Seq]
  `--> language_embeds:   [Seq,B,1024]

geo prompt / dummy prompt
  |
  v
geometry_encoder.pt
  |
  |--> geo_feats: [GeoSeq,B,256]
  `--> geo_masks: [B,GeoSeq]

text tokens + geo tokens
  |
  v
_encode_prompt
  |
  |--> prompt:      [PromptSeq,B,256]
  `--> prompt_mask: [B,PromptSeq]

prompt + visual tokens
  |
  v
transformer_encoder.pt
  |
  |--> memory:           [HW,B,256]
  |--> pos_embed:        [HW,B,256]
  |--> memory_text:      [PromptSeq,B,256]
  |--> spatial_shapes
  `--> valid_ratios

memory + object queries
  |
  v
transformer_decoder.pt
  |
  |--> hs:               [NumLayer,B,NumQuery,256]
  |--> presence_feats:   [B,1,256]
  `--> reference boxes

                      +---------------------------+
                      |                           |
                      v                           v
            dot_product_scoring.pt      segmentation_head.pt
                      |                           |
                      | pred_logits               | pred_masks
                      v                           v
                 scores / logits            mask logits
                      \                           /
                       \                         /
                        +-----------+-----------+
                                    |
                                    v
                           detector outputs
                           - pred_logits
                           - pred_boxes
                           - pred_boxes_xyxy
                           - pred_masks
                           - presence_logit_dec
```

实测单图链路：

- `prompt`: `[33,1,256]`
- `encoder_hidden_states`: `[5184,1,256]`
- `hs`: `[6,1,200,256]`
- `pred_logits`: `[1,200,1]`
- `pred_boxes`: `[1,200,4]`
- `pred_masks`: `[1,200,288,288]`

## Tracker Flow

下面这张图只看视频 tracker。

```text
detector outputs on current frame
  |
  |--> selected masks
  |--> selected scores
  `--> selected boxes
          |
          v
tracker_sam_heads.pt
  |
  |--> low_res_multimasks:  [B,3,288,288]
  |--> high_res_multimasks: [B,3,1008,1008]
  |--> ious:                [B,3]
  |--> low_res_masks:       [B,1,288,288]
  |--> high_res_masks:      [B,1,1008,1008]
  |--> obj_ptr:             [B,256]
  `--> object_score_logits: [B,1]
          |
          v
_encode_new_memory
  |
  |--> current_vision_feats[-1]:
  |    [HW,B,256] -> reshape -> [B,256,72,72]
  |
  |--> high_res_masks / object_score_logits
  v
tracker_maskmem_backbone.pt
  |
  |--> vision_features: [B,64,72,72]
  `--> vision_pos_enc:  [(B,64,72,72)]
          |
          v
memory prompt assembly
  |
  |--> mask memory tokens
  |--> object pointer tokens
  `--> temporal position tokens
          |
          v
tracker_transformer.pt
  |
  |--> memory:    [HW,B,256]
  |--> pos_embed: [HW,B,256]
  `--> padding_mask
          |
          v
propagated tracker state
  |
  v
video outputs per frame
  - out_obj_ids
  - out_probs
  - out_boxes_xywh
  - out_binary_masks
```

tracker 侧可以把模块边界理解成：

1. `tracker_sam_heads`
   从当前帧 tracker feature 生成 mask 和 object pointer

2. `tracker_maskmem_backbone`
   把 mask 写入 memory feature

3. `tracker_transformer`
   用 memory feature 和 pointer token 做时序传播

## What Is Saved

当前 `weights_modular/` 下的文件：

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

这些文件都是模块参数，不包含 Python 结构定义。
要恢复推理，必须用和原模块同构的 Python 类重新实例化，再 `load_state_dict(...)`。

## Assembly Structure

模块被分成两大块：

1. detector
   负责单帧文本检测与分割

2. tracker
   负责视频时序传播与 memory 更新

最终由 `build_video_model()` 组装成：

- `Sam3ImageOnVideoMultiGPU` detector
- `Sam3TrackerPredictor` tracker
- `Sam3VideoInferenceWithInstanceInteractivity` video model

## Detector Modules

### `vision_backbone.pt`

对应 Python 模块：
- `_create_vision_backbone(...)`

运行时角色：
- 提取图像多尺度视觉特征

输入：
- `image`: `torch.Tensor`
- 形状通常是 `[B, 3, 1008, 1008]`

输出：
- 不单独直接暴露给上层脚本，而是作为 `SAM3VLBackbone.visual`
- 经 `SAM3VLBackbone.forward_image(...)` 后得到：
  - `vision_features`: 最低分辨率主特征，通常 `[B, 256, 72, 72]`
  - `vision_pos_enc`: 多尺度位置编码列表
  - `backbone_fpn`: 多尺度特征列表，常见为
    - `[B, 256, 288, 288]`
    - `[B, 256, 144, 144]`
    - `[B, 256, 72, 72]`

谁消费它：
- `SAM3VLBackbone`
- `geometry_encoder`
- `transformer_encoder`
- `segmentation_head`

说明：
- 它本身只是视觉分支，不包含文本编码。

### `text_encoder.pt`

对应 Python 模块：
- `_create_text_encoder(...)`

运行时角色：
- 把文本 prompt 编码成 detector 使用的 token 特征

输入：
- `text`: `List[str]`
- `input_boxes`: 通常为 `None`
- `device`

输出：
- `text_attention_mask`: `[B, Seq]`
  - `False` 表示有效 token
  - `True` 表示 padding
- `text_memory_resized`: `[Seq, B, 256]`
- `inputs_embeds`: `[Seq, B, 1024]`

谁消费它：
- `SAM3VLBackbone.forward_text(...)`
- 后续 `transformer_encoder`
- 后续 `dot_product_scoring`

说明：
- 当前 detector 主链真正使用的是 `text_memory_resized` 和 `text_attention_mask`
- `inputs_embeds` 主要用于兼容和保留原始高维文本嵌入

### `geometry_encoder.pt`

对应 Python 模块：
- `_create_geometry_encoder(...)`

运行时角色：
- 把几何提示编码成 prompt token
- 支持 box / point / mask prompt

输入：
- `geo_prompt`
  - 包含 box / point / label / mask 等几何提示
- `img_feats`
  - 图像特征列表，主链里通常来自 `backbone_fpn`
- `img_sizes`
  - 每个特征层的 `(H, W)`
- `img_pos_embeds`
  - 对应特征层的位置编码

主链中常见输入语义：
- 文本检测时通常是“空几何提示”或 dummy prompt
- 交互式场景才会带真实点框提示

输出：
- `geo_feats`: `[PromptSeq, B, 256]`
- `geo_masks`: `[B, PromptSeq]`

当前单图文本检测实测：
- dummy 几何提示会产生 1 个 prompt token
- 文本 `"shoe"` 编码后是 32 个文本 token
- 所以 detector 主链里最终 `prompt` 实测为 `[33, 1, 256]`
- 对应 `prompt_mask` 实测为 `[1, 33]`

谁消费它：
- detector 的 `_encode_prompt(...)`
- 再进入 `transformer_encoder`

说明：
- 这个模块定义了“几何提示 token 化”的边界
- 后续如果你想把 box/point prompt 单独训练，这里就是核心接口

### `transformer_encoder.pt`

对应 Python 模块：
- `_create_sam3_transformer().encoder`
- 实际类型是 `TransformerEncoderFusion`

运行时角色：
- 融合图像 token 和 prompt token
- 输出供 decoder 使用的 memory

输入：
- `src`: `List[Tensor]`
  - 每个元素是 `[HW, B, 256]` 的视觉 token
- `prompt`: `[Seq, B, 256]`
- `src_key_padding_mask`: 通常为 `None`
- `src_pos`: 与 `src` 对应的位置编码列表
- `prompt_key_padding_mask`: `[B, Seq]`
- `prompt_pos`: `[Seq, B, 256]`
- `feat_sizes`: 特征图尺寸列表，如 `[(72, 72)]`

输出：
- 一个字典，关键字段包括：
  - `memory`: `[HW_total, B, 256]`
  - `padding_mask`
  - `pos_embed`: `[HW_total, B, 256]`
  - `memory_text`: `[Seq, B, 256]`
  - `level_start_index`
  - `spatial_shapes`
  - `valid_ratios`

谁消费它：
- `transformer_decoder`
- `segmentation_head`

说明：
- 这是 detector 里图文融合后的公共中间表示边界
- 单图文本检测实测：
  - `encoder_hidden_states`: `[5184, 1, 256]`
  - `pos_embed`: `[5184, 1, 256]`
  - `spatial_shapes`: `[1, 2]`
  - `valid_ratios`: `[1, 1, 2]`

### `transformer_decoder.pt`

对应 Python 模块：
- `_create_sam3_transformer().decoder`
- 实际类型是 `TransformerDecoder`

运行时角色：
- 用 object queries 从 encoder memory 中解码出候选目标

输入：
- `tgt`: `[NumQuery, B, 256]`
  - 一般来自 `query_embed.weight`
- `memory`: `[HW_total, B, 256]`
- `memory_key_padding_mask`
- `pos`: `[HW_total, B, 256]`
- `reference_boxes`: 通常为 `None`
- `level_start_index`
- `spatial_shapes`
- `valid_ratios`
- `tgt_mask`: 通常为 `None`
- `memory_text`: `[Seq, B, 256]`
- `text_attention_mask`: `[B, Seq]`
- `apply_dac`

输出：
- `hs`: `[NumLayer, B, NumQuery, 256]`
- `reference_boxes`
- `dec_presence_out`
- `dec_presence_feats`

谁消费它：
- detector `_update_scores_and_boxes(...)`
- `dot_product_scoring`
- `segmentation_head`

说明：
- 它本身不直接给你最终 boxes / scores
- 最终框和分数是 decoder 输出再经过 box head、presence head、dot-product scoring 组合出来的
- 当前 detector 主链实测：
  - `hs`: `[6, 1, 200, 256]`
  - `queries`: `[1, 200, 256]`
  - `pred_logits`: `[1, 200, 1]`
  - `pred_boxes`: `[1, 200, 4]`
  - `pred_boxes_xyxy`: `[1, 200, 4]`
  - `presence_logit_dec`: `[1, 1]`
  - `presence_feats`: `[1, 1, 256]`

### `dot_product_scoring.pt`

对应 Python 模块：
- `_create_dot_product_scoring()`

运行时角色：
- 计算 query 和文本 prompt 的匹配分数

输入：
- `hs`: `[NumLayer, B, NumQuery, 256]`
- `prompt`: `[Seq, B, 256]`
- `prompt_mask`: `[B, Seq]`

输出：
- `scores`: `[NumLayer, B, NumQuery, 1]`

谁消费它：
- detector `_update_scores_and_boxes(...)`
- 形成 `pred_logits`

说明：
- 这是文本匹配分数头，不负责 box，不负责 mask

### `segmentation_head.pt`

对应 Python 模块：
- `_create_segmentation_head()`
- 实际类型是 `UniversalSegmentationHead`

运行时角色：
- 把 decoder query 和图像像素特征变成实例 mask

输入：
- `backbone_feats`: 多尺度图像特征列表
- `obj_queries`: 通常是 decoder 输出的 query
- `image_ids`
- `encoder_hidden_states`: `[HW_total, B, 256]`
- `prompt`: `[Seq, B, 256]`
- `prompt_mask`: `[B, Seq]`

输出：
- 字典：
  - `pred_masks`
  - `semantic_seg`
  - `presence_logit`

主链真正关心：
- `pred_masks`

谁消费它：
- detector `_run_segmentation_heads(...)`
- 最终输出里的 `pred_masks`

说明：
- 它只负责 mask，不负责框回归

## Tracker Modules

### `tracker_maskmem_backbone.pt`

对应 Python 模块：
- `_create_tracker_maskmem_backbone()`
- 内核是 `SimpleMaskEncoder`

运行时角色：
- 把当前帧像素特征和 mask 编成 memory feature

输入：
- `pix_feat`: `[B, 256, 72, 72]`
- `masks`: `[B, 1, H, W]`

输出：
- 字典：
  - `vision_features`
  - `vision_pos_enc`

谁消费它：
- tracker memory bank
- `tracker_transformer`

说明：
- 这是 tracker 把 mask 写入时序记忆的入口
- 实测：
  - 输入 `pix_feat`: `[1, 256, 72, 72]`
  - 输入 `masks`: `[1, 1, 1008, 1008]`
  - 输出 `vision_features`: `[1, 64, 72, 72]`
  - 输出 `vision_pos_enc`: `[(1, 64, 72, 72)]`

### `tracker_transformer.pt`

对应 Python 模块：
- `_create_tracker_transformer()`
- 其核心是 `tracker.transformer.encoder`

运行时角色：
- 对当前帧特征和 memory prompt 做时序融合

输入：
- `src`: `[HW, B, 256]`
- `prompt`: `[PromptSeq, B, 64]`
- `src_mask`
- `prompt_mask`
- `src_key_padding_mask`
- `prompt_key_padding_mask`
- `src_pos`
- `prompt_pos`
- `feat_sizes`
- `num_obj_ptr_tokens`

输出：
- 字典：
  - `memory`: `[HW, B, 256]`
  - `pos_embed`: `[HW, B, 256]`
  - `padding_mask`

谁消费它：
- tracker 主推理流程

说明：
- 这是 tracker 时序传播的核心融合模块
- 实测：
  - `memory`: `[5184, 1, 256]`
  - `pos_embed`: `[5184, 1, 256]`
  - `padding_mask`: `None`

### `tracker_sam_heads.pt`

对应 Python 模块：
- `TrackerSamHeads(...)` 包起来的一组 tracker SAM 头

里面实际保存的是：
- `sam_prompt_encoder`
- `sam_mask_decoder`
- `obj_ptr_proj`
- `obj_ptr_tpos_proj`
- `mask_downsample`
- `maskmem_tpos_enc`
- `no_mem_embed`
- `no_mem_pos_enc`
- `no_obj_ptr`
- `no_obj_embed_spatial`

运行时角色：
- 根据 tracker 特征生成 mask、IoU、object pointer

输入：
- 不是一个单独稳定的“单函数模块接口”
- 它作为 tracker 内部一组头一起工作

输出：
- tracker 内部会得到：
  - low/high resolution masks
  - iou
  - object pointer
  - object score logits

谁消费它：
- `Sam3TrackerPredictor`

说明：
- 这个模块不是“单一 forward 头”，而是一组必须和 tracker 主体配套使用的参数集合
- 如果你后面要把 tracker 完全拆开，这一块通常要和 tracker runtime 一起迁出
- 通过 `tracker._forward_sam_heads(...)` 实测输出为：
  - `low_res_multimasks`: `[1, 3, 288, 288]`
  - `high_res_multimasks`: `[1, 3, 1008, 1008]`
  - `ious`: `[1, 3]`
  - `low_res_masks`: `[1, 1, 288, 288]`
  - `high_res_masks`: `[1, 1, 1008, 1008]`
  - `obj_ptr`: `[1, 256]`
  - `object_score_logits`: `[1, 1]`

## Runtime Boundaries

如果从“未来独立拆 runtime”的角度看，当前最稳定的边界是：

1. detector runtime
   - `vision_backbone`
   - `text_encoder`
   - `geometry_encoder`
   - `transformer_encoder`
   - `transformer_decoder`
   - `dot_product_scoring`
   - `segmentation_head`

2. tracker runtime
   - `tracker_maskmem_backbone`
   - `tracker_transformer`
   - `tracker_sam_heads`

3. assembly/runtime glue
   - prompt 打包
   - 图像预处理
   - box / mask 后处理
   - detector 输出到 tracker 输入的格式桥接

真正要完全脱离原仓库时，最重要的不是再多拆一个 `.pt` 文件，而是把第 3 部分单独固化下来。

## What Is Already Verified

当前这条非 JIT 模块化链路已验证过：

- `weights_modular/*.pt` 可正常加载
- assembled video model 可正常运行
- 与原始 `sam3.pt` 的视频推理结果已做过逐帧一致性验证

## Recommended Direction

如果你的目标是：

- 结构清楚
- 模块解耦
- 后续还能继续训练模块
- 未来再把推理 runtime 完全抽离

建议主线继续使用 `weights_modular`，不要把 JIT 当主方案。
