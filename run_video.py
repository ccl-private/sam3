from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import (
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT = ROOT / "sam3.pt"
VIDEO_PATH = ROOT / "assets" / "videos" / "bedroom.mp4"
OUT_DIR = ROOT / "runs"


def load_video_frames_for_vis(video_path: Path):
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
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def render_frame(frame_idx, video_frames, outputs_per_frame):
    plt.close("all")
    visualize_formatted_frame_output(
        frame_idx,
        video_frames,
        outputs_list=[outputs_per_frame],
        titles=["SAM3 video predictor"],
        figsize=(6, 4),
    )
    fig = plt.gcf()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = rgba.reshape(height, width, 4)[..., :3].copy()
    plt.close(fig)
    return img


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the video predictor.")

    gpus_to_use = list(range(torch.cuda.device_count()))
    predictor = build_sam3_video_predictor(
        gpus_to_use=gpus_to_use,
        checkpoint_path=str(DEFAULT_CKPT) if DEFAULT_CKPT.exists() else None,
    )

    video_frames_for_vis, fps = load_video_frames_for_vis(VIDEO_PATH)

    response = predictor.handle_request(
        request=dict(type="start_session", resource_path=str(VIDEO_PATH))
    )
    session_id = response["session_id"]

    _ = predictor.handle_request(
        request=dict(type="reset_session", session_id=session_id)
    )

    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text="person",
        )
    )
    _ = response["outputs"]

    outputs_per_frame = propagate_in_video(predictor, session_id)
    outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "bedroom_vis.mp4"

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
