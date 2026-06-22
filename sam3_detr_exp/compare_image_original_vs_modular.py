#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from sam3.visualization_utils import COLORS
from sam3_detr_exp.modular_pipeline import WEIGHTS_DIR, build_detector_model

EXP_ROOT = ROOT / "sam3_detr_exp"
DEFAULT_CKPT = ROOT / "sam3.pt"
DEFAULT_IMAGE = ROOT / "assets" / "images" / "test_image.jpg"
DEFAULT_BPE = ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
OUTPUT_DIR = EXP_ROOT / "outputs"

EXPECTED_MODULES = [
    "vision_backbone",
    "text_encoder",
    "transformer_encoder",
    "transformer_decoder",
    "segmentation_head",
    "geometry_encoder",
    "dot_product_scoring",
]


def assert_modular_weights_exist() -> None:
    missing = [name for name in EXPECTED_MODULES if not (WEIGHTS_DIR / f"{name}.pt").exists()]
    if missing:
        raise FileNotFoundError(
            "Missing modular weight files: "
            + ", ".join(missing)
            + f". Run {EXP_ROOT / 'run_video_det_modular.py'} first."
        )


def render_image_overlay(image: Image.Image, boxes, masks, scores, title: str) -> np.ndarray:
    canvas = np.array(image.convert("RGB"), copy=True)
    height, width = canvas.shape[:2]

    if masks is not None and len(masks) > 0:
        for idx in range(len(masks)):
            color = (COLORS[idx % len(COLORS)] * 255).astype(np.uint8)
            mask = masks[idx]
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()
            if mask.ndim == 3:
                mask = mask[0]
            mask_bool = mask > 0.5
            for channel in range(3):
                canvas[..., channel][mask_bool] = (
                    0.45 * color[channel] + 0.55 * canvas[..., channel][mask_bool]
                ).astype(np.uint8)

    if boxes is not None and len(boxes) > 0:
        for idx in range(len(boxes)):
            color = tuple(int(x) for x in (COLORS[idx % len(COLORS)] * 255))
            box = boxes[idx]
            if isinstance(box, torch.Tensor):
                box = box.detach().cpu().numpy()
            x0, y0, x1, y1 = [int(round(v)) for v in box.tolist()]
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
            score_text = ""
            if scores is not None and len(scores) > idx:
                score = scores[idx]
                if isinstance(score, torch.Tensor):
                    score = float(score.detach().cpu())
                score_text = f" {score:.3f}"
            cv2.putText(
                canvas,
                f"id={idx}{score_text}",
                (x0, max(y0 - 8, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    header_height = 40
    header = np.full((header_height, width, 3), 245, dtype=np.uint8)
    cv2.putText(
        header,
        title,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return np.concatenate([header, canvas], axis=0)


def build_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    divider = np.full((left.shape[0], 12, 3), 255, dtype=np.uint8)
    return np.concatenate([left, divider, right], axis=1)


def run_processor(processor: Sam3Processor, image: Image.Image, prompt: str):
    state = processor.set_image(image)
    state = processor.set_text_prompt(prompt, state)
    return state["boxes"], state["masks"], state["scores"]


def main() -> None:
    parser = ArgumentParser(description="Render original SAM3 vs modular weights on one image.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", type=str, default="shoe")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "image_original_vs_modular.png")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for image comparison.")
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    assert_modular_weights_exist()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    image = Image.open(args.image).convert("RGB")
    device = "cuda"

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        original_model = build_sam3_image_model(
            bpe_path=str(DEFAULT_BPE),
            device=device,
            eval_mode=True,
            checkpoint_path=str(args.checkpoint),
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
        )
        modular_model = build_detector_model(bpe_path=str(DEFAULT_BPE)).to(device).eval()

        original_processor = Sam3Processor(original_model, device=device)
        modular_processor = Sam3Processor(modular_model, device=device)

        original_boxes, original_masks, original_scores = run_processor(
            original_processor, image, args.prompt
        )
        modular_boxes, modular_masks, modular_scores = run_processor(
            modular_processor, image, args.prompt
        )

    left = render_image_overlay(
        image, original_boxes, original_masks, original_scores, "Original sam3.pt"
    )
    right = render_image_overlay(
        image, modular_boxes, modular_masks, modular_scores, "Modular weights_modular"
    )
    compare = build_side_by_side(left, right)

    cv2.imwrite(str(args.output), cv2.cvtColor(compare, cv2.COLOR_RGB2BGR))
    print(f"saved: {args.output}")
    print(
        "detections:",
        f"original={len(original_boxes)}",
        f"modular={len(modular_boxes)}",
    )


if __name__ == "__main__":
    main()
