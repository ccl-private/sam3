import argparse
from pathlib import Path

import torch
import torch.nn as nn

from sam3.model_builder import build_sam3_video_model

EXP_ROOT = Path(__file__).resolve().parent
SAM3_ROOT = EXP_ROOT.parent
WEIGHTS_DIR = EXP_ROOT / "weights_modular"

DEFAULT_CKPT = SAM3_ROOT / "sam3.pt"
BPE_PATH = SAM3_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"


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


def save_module_weights(video_model, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = video_model.detector
    tracker = video_model.tracker

    tracker_sam_heads = TrackerSamHeads(tracker)

    module_map = {
        "vision_backbone": detector.backbone.vision_backbone,
        "text_encoder": detector.backbone.language_backbone,
        "transformer_encoder": detector.transformer.encoder,
        "transformer_decoder": detector.transformer.decoder,
        "segmentation_head": detector.segmentation_head,
        "geometry_encoder": detector.geometry_encoder,
        "dot_product_scoring": detector.dot_prod_scoring,
        "tracker_sam_heads": tracker_sam_heads,
        "tracker_maskmem_backbone": tracker.maskmem_backbone,
        "tracker_transformer": tracker.transformer,
    }

    for name, module in module_map.items():
        path = out_dir / f"{name}.pt"
        torch.save(module.state_dict(), path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export modular SAM3 detector/tracker weights from a full sam3.pt checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CKPT,
        help=f"Path to the original SAM3 checkpoint. Default: {DEFAULT_CKPT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEIGHTS_DIR,
        help=f"Directory to save weights_modular/*.pt. Default: {WEIGHTS_DIR}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint not found: "
            f"{checkpoint_path}. "
            "Use --checkpoint /path/to/sam3.pt, or copy sam3.pt into the repo root."
        )

    video_model = build_sam3_video_model(
        checkpoint_path=str(checkpoint_path),
        load_from_HF=False,
        bpe_path=str(BPE_PATH),
        device=device,
        compile=False,
    )

    save_module_weights(video_model, output_dir)
    print("checkpoint:", checkpoint_path)
    print("saved:", output_dir)


if __name__ == "__main__":
    main()
