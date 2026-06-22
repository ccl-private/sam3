from pathlib import Path

import torch
import torch.nn as nn

from sam3.model.sam3_base_predictor import Sam3BasePredictor
from sam3.model.sam3_image import Sam3ImageOnVideoMultiGPU
from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor
from sam3.model.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity
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

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights_modular"
BPE_PATH = ROOT.parent / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"


def load_module_weights(module: nn.Module, name: str, weight_dir: Path = WEIGHTS_DIR) -> None:
    """Load one saved module weight file from WEIGHTS_DIR."""
    path = weight_dir / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing module weight file: {path}. Run run_video_det_modular.py first."
        )
    module.load_state_dict(torch.load(path, map_location="cpu"))


class TrackerSamHeads(nn.Module):
    def __init__(self, tracker: Sam3TrackerPredictor):
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


def build_detector_modules(bpe_path: str = str(BPE_PATH)) -> dict:
    """Build detector submodules and load weights for each block."""
    vision_backbone = _create_vision_backbone(compile_mode=None, enable_inst_interactivity=True)
    load_module_weights(vision_backbone, "vision_backbone")

    text_encoder = _create_text_encoder(bpe_path)
    load_module_weights(text_encoder, "text_encoder")

    backbone = SAM3VLBackbone(visual=vision_backbone, text=text_encoder, scalp=1)

    transformer = _create_sam3_transformer()
    load_module_weights(transformer.encoder, "transformer_encoder")
    load_module_weights(transformer.decoder, "transformer_decoder")

    segmentation_head = _create_segmentation_head()
    load_module_weights(segmentation_head, "segmentation_head")

    geometry_encoder = _create_geometry_encoder()
    load_module_weights(geometry_encoder, "geometry_encoder")

    dot_prod_scoring = _create_dot_product_scoring()
    load_module_weights(dot_prod_scoring, "dot_product_scoring")

    return {
        "vision_backbone": vision_backbone,
        "text_encoder": text_encoder,
        "backbone": backbone,
        "transformer": transformer,
        "segmentation_head": segmentation_head,
        "geometry_encoder": geometry_encoder,
        "dot_prod_scoring": dot_prod_scoring,
    }


def build_detector_model(bpe_path: str = str(BPE_PATH)) -> Sam3ImageOnVideoMultiGPU:
    """Build the detector model from separate detector modules."""
    det_modules = build_detector_modules(bpe_path=bpe_path)
    main_dot_prod_scoring = _create_dot_product_scoring()
    main_dot_prod_scoring.load_state_dict(det_modules["dot_prod_scoring"].state_dict())

    detector = Sam3ImageOnVideoMultiGPU(
        num_feature_levels=1,
        backbone=det_modules["backbone"],
        transformer=det_modules["transformer"],
        segmentation_head=det_modules["segmentation_head"],
        semantic_segmentation_head=None,
        input_geometry_encoder=det_modules["geometry_encoder"],
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=main_dot_prod_scoring,
        supervise_joint_box_scores=True,
    )
    return detector


def build_tracker_modules() -> Sam3TrackerPredictor:
    """Build tracker submodules and load weights for each block."""
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

    tracker_sam_heads = TrackerSamHeads(tracker)
    load_module_weights(tracker_sam_heads, "tracker_sam_heads")
    load_module_weights(tracker.maskmem_backbone, "tracker_maskmem_backbone")
    load_module_weights(tracker.transformer, "tracker_transformer")

    return tracker


def build_tracker_model() -> Sam3TrackerPredictor:
    """Build the tracker model with its own modular subcomponents."""
    return build_tracker_modules()


def build_video_model(device: str = "cuda") -> Sam3VideoInferenceWithInstanceInteractivity:
    """Assemble the full video model from detector + tracker modules."""
    detector = build_detector_model()
    tracker = build_tracker_model()

    model = Sam3VideoInferenceWithInstanceInteractivity(
        detector=detector,
        tracker=tracker,
        score_threshold_detection=0.5,
        assoc_iou_thresh=0.1,
        det_nms_thresh=0.1,
        new_det_thresh=0.7,
        hotstart_delay=15,
        hotstart_unmatch_thresh=8,
        hotstart_dup_thresh=8,
        suppress_unmatched_only_within_hotstart=True,
        min_trk_keep_alive=-1,
        max_trk_keep_alive=30,
        init_trk_keep_alive=30,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
        suppress_det_close_to_boundary=False,
        fill_hole_area=16,
        recondition_every_nth_frame=16,
        masklet_confirmation_enable=False,
        decrease_trk_keep_alive_for_empty_masklets=False,
        image_size=1008,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        compile_model=False,
    )

    model.to(device=device)
    model.eval()
    return model


class ModularVideoPredictor(Sam3BasePredictor):
    """A small wrapper around the assembled video model."""

    def __init__(self, model, async_loading_frames: bool = False, video_loader_type: str = "cv2", default_output_prob_thresh: float = 0.5):
        super().__init__()
        self.model = model
        self._all_inference_states = {}
        self.async_loading_frames = async_loading_frames
        self.video_loader_type = video_loader_type
        self.default_output_prob_thresh = default_output_prob_thresh


__all__ = [
    "WEIGHTS_DIR",
    "BPE_PATH",
    "load_module_weights",
    "TrackerSamHeads",
    "build_detector_modules",
    "build_detector_model",
    "build_tracker_modules",
    "build_tracker_model",
    "build_video_model",
    "ModularVideoPredictor",
]
