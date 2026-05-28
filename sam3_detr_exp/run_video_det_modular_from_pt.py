from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from sam3.model.sam3_base_predictor import Sam3BasePredictor
from sam3.model.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity
from sam3.model.sam3_image import Sam3ImageOnVideoMultiGPU
from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor
from sam3.model.vl_combiner import SAM3VLBackbone
from sam3.model_builder import (
    _create_dot_product_scoring,
    _create_geometry_encoder,
    _create_segmentation_head,
    _create_sam3_transformer,
    _create_tracker_maskmem_backbone,
    _create_tracker_transformer,
    _create_text_encoder,
    _create_vision_backbone,
)
from sam3.visualization_utils import (
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)

EXP_ROOT = Path(__file__).resolve().parent
SAM3_ROOT = EXP_ROOT.parent
RUNS_DIR = EXP_ROOT / "runs"
WEIGHTS_DIR = EXP_ROOT / "weights_modular"

BPE_PATH = SAM3_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
VIDEO_PATH = SAM3_ROOT / "assets" / "videos" / "bedroom.mp4"
# VIDEO_PATH = Path("/mnt/mnt108_hdd/zjg/video/test3.mp4")
PROMPT_TEXT = "person"
CONFIDENCE_THRESHOLD = 0.5

# ============================================================================
# 模块化 PT 说明 (基于 1008 输入分辨率的视频推理):
# 说明补充 (Notes):
# - image_size=1008, backbone_stride=14; low-res mask size=288 (1008/14*4)
# - text prompt 全局生效，但 add_prompt 只在指定帧触发推理
# - score_threshold_detection/new_det_thresh/hotstart_* 会影响是否输出目标
# - feature_cache 会缓存文本与 backbone 特征，影响后续帧推理
#
# Detector 模块 (用于每帧目标检测):
# - vision_backbone: 输入 [1, 3, 1008, 1008]
#   输出 backbone_fpn/vision_pos_enc 为 3 个尺度:
#   [1,256,288,288], [1,256,144,144], [1,256,72,72]
#   vision_features 取最低分辨率 [1,256,72,72]
#   vision_pos_enc 由 PositionEmbeddingSine 计算 (sin/cos)
#   输入: frame tensor [B,3,1008,1008]
#   输出: FPN feats/pos_enc (3 scales)
#
# - text_encoder: 输入 1 条文本
#   输出 language_features [32,1,256], language_mask [1,32]
#   language_embeds [32,1,1024]
#   输入: text tokens (len=32)
#   文本先经 BPE 分词，"person" 拆成 32 个 token
#
# - vl_backbone: 组合视觉+文本，输出 backbone_out 字典
#   输出: backbone_out (含视觉/文本特征)
#
# - transformer: 输入图像特征 + 文本特征 + 几何提示
#   输出 pred_boxes [1,200,4], pred_logits [1,200,1], pred_masks [1,200,288,288]
#   同时输出 queries [1,200,256] 与 presence_feats [1,1,256]
#   输入: FPN feats + text feats + geometry prompt
#
# - segmentation_head: 解码为低分辨率 mask logits
#   输入: queries + pixel features
#   输出: pred_masks [1,200,288,288]
#
# - geometry_encoder: 编码点/框/掩码提示为 prompt tokens
#   输入: points/boxes/masks
#   输出: prompt tokens
#
# - dot_product_scoring: 查询-文本匹配分数 (对应 pred_logits)
#   输入: queries + text feats
#   输出: pred_logits [1,200,1]
#
# Tracker 模块 (用于视频跟踪):
# - tracker_sam_heads: SAM heads + pointer/tpos 相关参数
#   sam_prompt_encoder / sam_mask_decoder / obj_ptr_proj / obj_ptr_tpos_proj
#   mask_downsample / maskmem_tpos_enc / no_mem_* / no_obj_*
#   输入: prompt/mask
#   输出: init masklet + pointer
# - tracker_maskmem_backbone: SimpleMaskEncoder (64-dim memory encoding)
#   输入 mask [H,W]，输出 memory features [64,72,72]
#   包含: PositionEmbeddingSine(64-dim) + SimpleMaskDownSampler + CXBlock + SimpleFuser
#   输入: masklet [H,W]
#   输出: memory bank features [64,72,72]
#
# - tracker_transformer: RoPEAttention + TransformerDecoderLayerv2 (4 layers)
#   self_attention: RoPEAttention(256-dim, 1 head, feat_sizes=[72,72])
#   cross_attention: RoPEAttention(256-dim, 1 head, kv_in_dim=64, feat_sizes=[72,72])
#   encoder: TransformerEncoderCrossAttention(4 layers, d_model=256)
#   wrapper: TransformerWrapper(encoder, decoder=None, d_model=256)
#   输入: current frame feats + memory bank feats
#   输出: updated masklet logits
#
# - Sam3TrackerPredictor: 封装 SAM decoder + memory management
#   num_maskmem=7 (保留 7 帧记忆), max_cond_frames_in_attn=4
#   multimask_output_for_tracking=True (输出 3 个候选 mask)
#   use_memory_selection=True (时间消歧义)
#   输出: out_binary_masks / out_obj_ids / out_probs
#
# 视频跟踪流程:
# 1. start_session: 初始化视频状态，加载视频帧
# 2. add_prompt(frame_index=0, text="person"): 第 0 帧检测目标
#    - detector 输出 boxes [N,4], masks [N,1,H,W], scores [N]
#    - tracker 初始化 memory state (7 帧记忆窗口)
# 3. propagate_in_video: 逐帧传播跟踪
#    - 未匹配帧: detector 重新检测
#    - 已跟踪帧: tracker 用 memory 预测 mask (更快更稳定)
#    - 输出 obj_id_to_mask, obj_id_to_score
#
# 模块配合与论文名词对应:
# - detector 负责发现/重检目标, tracker 负责跨帧传播与稳定输出
# - memory bank 对应 tracker_maskmem_backbone + tracker_transformer 的记忆编码/注意力
# - masklet 是每个 obj_id 的时序 mask, 输出在 out_binary_masks / obj_id_to_mask
#
# 帧级流程 (Frame-wise):
# 1) 第 0 帧 (有 prompt):
#    - detector 运行 -> det masks [N,288,288] / boxes [N,4] / scores [N]
#    - tracker_sam_heads 用 det masks 初始化 obj_id/masklet/pointer
#    - tracker_maskmem_backbone 编码 memory bank [64,72,72]
#    - tracker_transformer 写入/准备跨帧传播
#    - 输出第 0 帧结果 out_binary_masks [N,H,W]
# 2) 后续帧:
#    - detector 每帧都会跑 backbone/text/transformer 得到 det masks/boxes/scores
#    - 仅在有 text/geometry prompt 时允许引入新检测 (allow_new_detections)
#    - tracker_transformer 基于 memory bank 传播并与 det 做关联/更新
#    - 生成本帧 out_binary_masks [N,H,W] / out_obj_ids [N] / out_probs [N]
# ============================================================================


def load_module_weights(module, name: str) -> None:
    path = WEIGHTS_DIR / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run run_video_det_modular.py to export module weights."
        )
    module.load_state_dict(torch.load(path, map_location="cpu"))


class TrackerSamHeads(nn.Module):
    def __init__(self, tracker):
        super().__init__()
        self.sam_prompt_encoder = tracker.sam_prompt_encoder
        self.sam_mask_decoder = tracker.sam_mask_decoder
        self.obj_ptr_proj = tracker.obj_ptr_proj
        self.obj_ptr_tpos_proj = tracker.obj_ptr_tpos_proj
        self.mask_downsample = tracker.mask_downsample
        self.maskmem_tpos_enc = tracker.maskmem_tpos_enc
        self.no_mem_embed = tracker.no_mem_embed
        self.no_mem_pos_enc = tracker.no_mem_pos_enc
        self.no_obj_ptr = tracker.no_obj_ptr
        self.no_obj_embed_spatial = tracker.no_obj_embed_spatial


def build_detector_modules():
    """逐模块构建 detector，不依赖任何 checkpoint。
    
    Detector 负责每帧的目标检测 (text-guided grounding)。
    流程: vision_backbone -> text_encoder -> vl_backbone -> transformer -> segmentation_head
    """
    # 1. vision_backbone: ViT + neck
    #    输入 [1,3,1008,1008] -> 输出 vision_features [1,256,72,72] + 3 尺度 FPN
    vision_backbone = _create_vision_backbone(compile_mode=None, enable_inst_interactivity=True)
    load_module_weights(vision_backbone, "vision_backbone")

    # 2. text_encoder: BPE tokenizer + VETextEncoder
    #    输入 "person" -> BPE tokenization (32 tokens) -> language_features [32,1,256]
    text_encoder = _create_text_encoder(str(BPE_PATH))
    load_module_weights(text_encoder, "text_encoder")

    # 3. vl_backbone: 组合视觉+文本 (无融合，仅包装)
    backbone = SAM3VLBackbone(visual=vision_backbone, text=text_encoder, scalp=1)

    # 4. transformer: 6-layer encoder + 6-layer decoder
    #    论文图示里我们把 transformer 拆成 encoder/decoder 两个模块
    transformer = _create_sam3_transformer()
    load_module_weights(transformer.encoder, "transformer_encoder")
    load_module_weights(transformer.decoder, "transformer_decoder")

    # 5. segmentation_head: pixel decoder + universal seg head
    #    将 transformer queries 解码为 mask logits
    segmentation_head = _create_segmentation_head()
    load_module_weights(segmentation_head, "segmentation_head")

    # 6. geometry_encoder: 编码点/框/掩码提示为 prompt tokens
    geometry_encoder = _create_geometry_encoder()
    load_module_weights(geometry_encoder, "geometry_encoder")

    # 7. dot_product_scoring: 查询-文本匹配分数
    #    计算 transformer queries 与 text embeddings 的点积
    dot_prod_scoring = _create_dot_product_scoring()
    load_module_weights(dot_prod_scoring, "dot_product_scoring")

    return {
        "backbone": backbone,
        "transformer": transformer,
        "segmentation_head": segmentation_head,
        "geometry_encoder": geometry_encoder,
        "dot_prod_scoring": dot_prod_scoring,
    }


def build_tracker_modules():
    """构建 tracker 并加载论文图示拆分的 tracker 子模块权重。

    Tracker 由两层结构组成:
    1) SAM heads: 提供 prompt 编码 + mask 解码 + pointer/tpos 投影
        - sam_prompt_encoder / sam_mask_decoder
        - obj_ptr_proj / obj_ptr_tpos_proj
        - mask_downsample / maskmem_tpos_enc / no_mem_* / no_obj_*
    2) Tracking backbone:
        - tracker_maskmem_backbone: 64-dim memory encoder
        - tracker_transformer: memory attention transformer
    """
    maskmem_backbone = _create_tracker_maskmem_backbone()
    transformer = _create_tracker_transformer()
    tracker = Sam3TrackerPredictor(
        image_size=1008,
        num_maskmem=7,
        backbone=None,
        backbone_stride=14,
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        multimask_output_in_sam=True,
        forward_backbone_per_frame_for_eval=True,
        trim_past_non_cond_mem_for_eval=False,
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        always_start_from_first_ann_frame=False,
        non_overlap_masks_for_mem_enc=False,
        non_overlap_masks_for_output=False,
        max_cond_frames_in_attn=4,
        offload_output_to_cpu_for_eval=False,
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        clear_non_cond_mem_around_input=True,
        fill_hole_area=0,
        use_memory_selection=True,
    )

    # 1) SAM heads: 负责从 prompt 生成初始 mask + 目标指针
    tracker_sam_heads = TrackerSamHeads(tracker)
    load_module_weights(tracker_sam_heads, "tracker_sam_heads")

    # 2) memory encoder + transformer: 负责跨帧记忆与传播
    load_module_weights(tracker.maskmem_backbone, "tracker_maskmem_backbone")
    load_module_weights(tracker.transformer, "tracker_transformer")
    return tracker


def build_video_model(device: str):
    """组装完整的视频推理模型 (detector + tracker)，不依赖 checkpoint。
    
    组装流程:
    1. 构建 detector (Sam3ImageOnVideoMultiGPU): 用于每帧检测
    2. 构建 tracker (Sam3TrackerPredictor): 用于帧间跟踪
    3. 组装 Sam3VideoInferenceWithInstanceInteractivity: 统一管理检测+跟踪
    """
    det_modules = build_detector_modules()

    # 构建 detector (Sam3ImageOnVideoMultiGPU)
    #    注意: dot_prod_scoring 需要重新创建 MLP (因为 state_dict 已加载)
    main_dot_prod_scoring = _create_dot_product_scoring()
    main_dot_prod_scoring.load_state_dict(det_modules["dot_prod_scoring"].state_dict())

    detector = Sam3ImageOnVideoMultiGPU(
        num_feature_levels=1, backbone=det_modules["backbone"],
        transformer=det_modules["transformer"],
        segmentation_head=det_modules["segmentation_head"],
        semantic_segmentation_head=None,
        input_geometry_encoder=det_modules["geometry_encoder"],
        use_early_fusion=True, use_dot_prod_scoring=True,
        dot_prod_scoring=main_dot_prod_scoring,
        supervise_joint_box_scores=True,
    )

    # 构建 tracker
    tracker = build_tracker_modules()

    # 组装视频推理模型 (Sam3VideoInferenceWithInstanceInteractivity)
    #    核心逻辑:
    #    - score_threshold_detection=0.5: 检测分数阈值
    #    - assoc_iou_thresh=0.1: 跟踪关联 IoU 阈值
    #    - det_nms_thresh=0.1: 检测 NMS 阈值
    #    - new_det_thresh=0.7: 新目标检测阈值 (高于此分数视为新目标)
    #    - hotstart_delay=15: 热启动延迟 (前 15 帧强制检测)
    #    - hotstart_unmatch_thresh=8: 未匹配帧数阈值 (超过则重新检测)
    #    - hotstart_dup_thresh=8: 重复检测阈值
    #    - suppress_overlapping_based_on_recent_occlusion_threshold=0.7: 抑制重叠 mask
    #    - fill_hole_area=16: 填充 mask 孔洞
    #    - recondition_every_nth_frame=16: 每 16 帧重新校准
    model = Sam3VideoInferenceWithInstanceInteractivity(
        detector=detector, tracker=tracker,
        score_threshold_detection=0.5, assoc_iou_thresh=0.1,
        det_nms_thresh=0.1, new_det_thresh=0.7, hotstart_delay=15,
        hotstart_unmatch_thresh=8, hotstart_dup_thresh=8,
        suppress_unmatched_only_within_hotstart=True,
        min_trk_keep_alive=-1, max_trk_keep_alive=30, init_trk_keep_alive=30,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
        suppress_det_close_to_boundary=False, fill_hole_area=16,
        recondition_every_nth_frame=16, masklet_confirmation_enable=False,
        decrease_trk_keep_alive_for_empty_masklets=False, image_size=1008,
        image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5),
        compile_model=False,
    )

    model.to(device=device)
    model.eval()
    return model


class ModularVideoPredictor(Sam3BasePredictor):
    """基于模块化权重的视频预测器，复用 Sam3BasePredictor 的 API。
    
    提供与 build_sam3_video_predictor 相同的接口:
    - start_session: 初始化视频会话
    - add_prompt: 添加文本/点/框提示
    - propagate_in_video: 逐帧传播跟踪
    - reset_session: 重置会话
    - close_session: 关闭会话
    """
    def __init__(
        self, model, async_loading_frames: bool = False,
        video_loader_type: str = "cv2",
        default_output_prob_thresh: float = CONFIDENCE_THRESHOLD,
    ):
        super().__init__()
        self.model = model
        self._all_inference_states = {}
        self.async_loading_frames = async_loading_frames
        self.video_loader_type = video_loader_type
        self.default_output_prob_thresh = default_output_prob_thresh


def load_video_frames_for_vis(video_path: Path):
    """加载视频帧用于可视化。"""
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, float(fps)


def propagate_in_video(predictor, session_id):
    """逐帧传播跟踪，收集每帧的输出。
    
    返回: {frame_index: {obj_id_to_mask, obj_id_to_score, ...}}
    """
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request={"type": "propagate_in_video", "session_id": session_id}
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def render_frame(frame_idx, video_frames, outputs_per_frame):
    """渲染单帧可视化结果。"""
    plt.close("all")
    visualize_formatted_frame_output(
        frame_idx, video_frames, outputs_list=[outputs_per_frame],
        titles=["SAM3 modular video predictor"], figsize=(6, 4),
    )
    fig = plt.gcf()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = rgba.reshape(height, width, 4)[..., :3].copy()
    plt.close(fig)
    return img


def main() -> None:
    """主函数: 加载模块化权重 -> 构建视频模型 -> 执行跟踪 -> 保存结果。"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the video tracker.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 1. 构建视频模型 (detector + tracker)
    device = "cuda"
    model = build_video_model(device)
    predictor = ModularVideoPredictor(model)

    # 2. 加载视频帧
    video_frames_for_vis, fps = load_video_frames_for_vis(VIDEO_PATH)

    # 3. 初始化会话
    response = predictor.handle_request(
        request={"type": "start_session", "resource_path": str(VIDEO_PATH)}
    )
    session_id = response["session_id"]

    # 4. 重置会话 (清除之前的状态)
    _ = predictor.handle_request(
        request={"type": "reset_session", "session_id": session_id}
    )

    # 5. 添加提示: 在第 0 帧用文本 "person" 检测目标
    #    流程: detector 输出 boxes/masks/scores -> tracker 初始化 memory
    response = predictor.handle_request(
        request={
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": PROMPT_TEXT,
        }
    )
    _ = response["outputs"]

    # 6. 逐帧传播跟踪
    #    流程: 
    #    - 已跟踪帧: tracker 用 memory 预测 mask (快速)
    #    - 未匹配帧: detector 重新检测 (稳定)
    #    - 输出: obj_id_to_mask, obj_id_to_score
    outputs_per_frame = propagate_in_video(predictor, session_id)
    outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)

    # 7. 保存可视化结果
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / "bedroom_vis_modular_tracked_from_pt.mp4"

    first_img = render_frame(0, video_frames_for_vis, outputs_per_frame)
    height, width = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    writer.write(cv2.cvtColor(first_img, cv2.COLOR_RGB2BGR))

    for frame_idx in range(1, len(video_frames_for_vis)):
        img = render_frame(frame_idx, video_frames_for_vis, outputs_per_frame)
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    writer.release()
    print("saved:", out_path)


if __name__ == "__main__":
    main()
