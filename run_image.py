from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt
import torch

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT = ROOT / "sam3.pt"
FALLBACK_CKPT = Path("/home/jx/.cache/modelscope/hub/models/facebook/sam3/sam3.pt")
IMAGE_PATH = ROOT / "assets" / "images" / "test_image.jpg"
BPE_PATH = ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
RUNS_DIR = ROOT / "runs"


def resolve_checkpoint() -> Path:
    if DEFAULT_CKPT.exists():
        return DEFAULT_CKPT
    if FALLBACK_CKPT.exists():
        return FALLBACK_CKPT
    raise FileNotFoundError(
        "Checkpoint not found. Copy sam3.pt into the repo root or update the path."
    )


def visualize(image: Image.Image, masks, boxes, scores, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)

    if masks is not None and len(masks) > 0:
        mask = masks[0].detach().cpu().numpy()
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        ax.imshow(mask, alpha=0.5, cmap="jet")

    if boxes is not None and len(boxes) > 0:
        box = boxes[0].detach().cpu().numpy()
        x0, y0, x1, y1 = box.tolist()
        rect = plt.Rectangle(
            (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="lime", linewidth=2
        )
        ax.add_patch(rect)
        if scores is not None and len(scores) > 0:
            ax.text(
                x0,
                y0,
                f"{scores[0].item():.3f}",
                color="lime",
                fontsize=10,
                bbox=dict(facecolor="black", alpha=0.4, pad=2, edgecolor="none"),
            )

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    ckpt_path = resolve_checkpoint()
    image = Image.open(IMAGE_PATH).convert("RGB")

    model = build_sam3_image_model(
        bpe_path=str(BPE_PATH),
        checkpoint_path=str(ckpt_path),
        device=device,
        eval_mode=True,
    )
    processor = Sam3Processor(model, device=device, confidence_threshold=0.5)

    state = processor.set_image(image)
    out = processor.set_text_prompt(state=state, prompt="shoe")

    masks, boxes, scores = out["masks"], out["boxes"], out["scores"]
    print("masks:", masks.shape)
    print("boxes:", boxes.shape)
    print("scores:", scores.shape)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / "image_vis.png"
    visualize(image, masks, boxes, scores, out_path)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
