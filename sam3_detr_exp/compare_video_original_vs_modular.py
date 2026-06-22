#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sam3.model.sam3_video_predictor import Sam3VideoPredictorMultiGPU
from sam3.visualization_utils import render_masklet_frame
from sam3_detr_exp.modular_pipeline import WEIGHTS_DIR, ModularVideoPredictor, build_video_model

EXP_ROOT = ROOT / "sam3_detr_exp"
DEFAULT_CKPT = ROOT / "sam3.pt"
DEFAULT_VIDEO = ROOT / "assets" / "videos" / "bedroom.mp4"
OUTPUT_DIR = EXP_ROOT / "outputs"
DEFAULT_BPE = ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"

EXPECTED_MODULES = [
    "vision_backbone",
    "text_encoder",
    "transformer_encoder",
    "transformer_decoder",
    "segmentation_head",
    "geometry_encoder",
    "dot_product_scoring",
    "tracker_sam_heads",
    "tracker_maskmem_backbone",
    "tracker_transformer",
]


def assert_modular_weights_exist() -> None:
    missing = [name for name in EXPECTED_MODULES if not (WEIGHTS_DIR / f"{name}.pt").exists()]
    if missing:
        raise FileNotFoundError(
            "Missing modular weight files: "
            + ", ".join(missing)
            + f". Run {EXP_ROOT / 'run_video_det_modular.py'} first."
        )


def load_video_frames(video_path: Path):
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


def collect_predictor_outputs(predictor, video_path: Path, prompt: str, max_frames: int | None):
    session_id = predictor.handle_request(
        {"type": "start_session", "resource_path": str(video_path)}
    )["session_id"]
    predictor.handle_request({"type": "reset_session", "session_id": session_id})
    predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": prompt,
        }
    )

    request = {
        "type": "propagate_in_video",
        "session_id": session_id,
        "propagation_direction": "forward",
    }
    if max_frames is not None:
        request["max_frame_num_to_track"] = max_frames

    outputs = {}
    for response in predictor.handle_stream_request(request):
        outputs[response["frame_index"]] = response["outputs"]

    predictor.handle_request({"type": "close_session", "session_id": session_id})
    return outputs


def add_title(frame: np.ndarray, title: str) -> np.ndarray:
    header = np.full((40, frame.shape[1], 3), 245, dtype=np.uint8)
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
    return np.concatenate([header, frame], axis=0)


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    divider = np.full((left.shape[0], 12, 3), 255, dtype=np.uint8)
    return np.concatenate([left, divider, right], axis=1)


def main() -> None:
    parser = ArgumentParser(description="Render original SAM3 vs modular weights on one video.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--prompt", type=str, default="person")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "video_original_vs_modular.mp4")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for video comparison.")
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    assert_modular_weights_exist()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    frames, fps = load_video_frames(args.video)

    original_predictor = Sam3VideoPredictorMultiGPU(
        checkpoint_path=str(args.checkpoint),
        bpe_path=str(DEFAULT_BPE),
        gpus_to_use=[0],
    )
    original_outputs = collect_predictor_outputs(
        original_predictor, args.video, args.prompt, args.max_frames
    )
    del original_predictor
    torch.cuda.empty_cache()

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        modular_model = build_video_model(device="cuda")
    modular_predictor = ModularVideoPredictor(modular_model)
    modular_outputs = collect_predictor_outputs(
        modular_predictor, args.video, args.prompt, args.max_frames
    )

    frame_indices = sorted(set(original_outputs) & set(modular_outputs))
    if not frame_indices:
        raise RuntimeError("No overlapping output frames were produced.")

    first_idx = frame_indices[0]
    left = add_title(
        render_masklet_frame(frames[first_idx], original_outputs[first_idx], frame_idx=first_idx),
        "Original sam3.pt",
    )
    right = add_title(
        render_masklet_frame(frames[first_idx], modular_outputs[first_idx], frame_idx=first_idx),
        "Modular weights_modular",
    )
    first_compare = side_by_side(left, right)

    height, width = first_compare.shape[:2]
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    writer.write(cv2.cvtColor(first_compare, cv2.COLOR_RGB2BGR))

    preview_path = args.output.with_suffix(".png")
    cv2.imwrite(str(preview_path), cv2.cvtColor(first_compare, cv2.COLOR_RGB2BGR))

    for frame_idx in frame_indices[1:]:
        left = add_title(
            render_masklet_frame(frames[frame_idx], original_outputs[frame_idx], frame_idx=frame_idx),
            "Original sam3.pt",
        )
        right = add_title(
            render_masklet_frame(frames[frame_idx], modular_outputs[frame_idx], frame_idx=frame_idx),
            "Modular weights_modular",
        )
        compare = side_by_side(left, right)
        writer.write(cv2.cvtColor(compare, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"saved: {args.output}")
    print(f"saved: {preview_path}")
    print("frames:", len(frame_indices))


if __name__ == "__main__":
    main()
