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
from sam3.visualization_utils import COLORS
from sam3_detr_exp.modular_pipeline import WEIGHTS_DIR, build_detector_model

EXP_ROOT = ROOT / "sam3_detr_exp"
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


def parse_box(box_str: str):
    parts = [float(x.strip()) for x in box_str.split(",")]
    if len(parts) != 4:
        raise ValueError("--box must be x0,y0,x1,y1")
    x0, y0, x1, y1 = parts
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Expected x1>x0 and y1>y0")
    return x0, y0, x1, y1


def pixel_xyxy_to_normalized_cxcywh(box_xyxy, image_width: int, image_height: int):
    x0, y0, x1, y1 = box_xyxy
    cx = ((x0 + x1) * 0.5) / image_width
    cy = ((y0 + y1) * 0.5) / image_height
    w = (x1 - x0) / image_width
    h = (y1 - y0) / image_height
    return [cx, cy, w, h]


def render_overlay(image: Image.Image, boxes, masks, scores, title: str) -> np.ndarray:
    canvas = np.array(image.convert("RGB"), copy=True)

    if masks is not None and len(masks) > 0:
        for idx in range(len(masks)):
            color = (COLORS[idx % len(COLORS)] * 255).astype(np.uint8)
            mask = masks[idx]
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()
            if mask.ndim == 3:
                mask = mask[0]
            mask_bool = mask > 0.5
            for c in range(3):
                canvas[..., c][mask_bool] = (
                    0.45 * color[c] + 0.55 * canvas[..., c][mask_bool]
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

    header = np.full((40, canvas.shape[1], 3), 245, dtype=np.uint8)
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


def main() -> None:
    parser = ArgumentParser(
        description="Run modular DETR-only inference with either text prompt or box prompt."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--text", type=str, default=None, help="Text prompt, e.g. 'shoe'")
    parser.add_argument(
        "--box",
        type=str,
        default=None,
        help="Pixel box prompt in x0,y0,x1,y1 format, e.g. 120,80,360,420",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "detr_prompt_inference.png",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if (args.text is None) == (args.box is None):
        raise ValueError("Use exactly one of --text or --box")

    assert_modular_weights_exist()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    image = Image.open(args.image).convert("RGB")
    image_w, image_h = image.size
    device = "cuda"

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        model = build_detector_model(bpe_path=str(DEFAULT_BPE)).to(device).eval()
        processor = Sam3Processor(
            model,
            device=device,
            confidence_threshold=args.threshold,
        )

        state = processor.set_image(image)
        if args.text is not None:
            state = processor.set_text_prompt(args.text, state)
            title = f"Modular DETR text prompt: {args.text}"
        else:
            prompt_box = parse_box(args.box)
            normalized_box = pixel_xyxy_to_normalized_cxcywh(prompt_box, image_w, image_h)
            state = processor.add_geometric_prompt(normalized_box, True, state)
            title = f"Modular DETR box prompt: {args.box}"

    overlay = render_overlay(image, state["boxes"], state["masks"], state["scores"], title)
    cv2.imwrite(str(args.output), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"saved: {args.output}")
    print("detections:", len(state["boxes"]))
    print("boxes shape:", tuple(state["boxes"].shape))
    print("masks shape:", tuple(state["masks"].shape))
    print("scores shape:", tuple(state["scores"].shape))


if __name__ == "__main__":
    main()
