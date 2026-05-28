from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.transforms import v2

from sam3.model import box_ops
from sam3.model.data_misc import FindStage, interpolate
from sam3.model_builder import (
    _create_dot_product_scoring,
    _create_geometry_encoder,
    _create_sam3_model,
    _create_sam3_transformer,
    _create_segmentation_head,
    _create_text_encoder,
    _create_vl_backbone,
    _create_vision_backbone,
)
from sam3.visualization_utils import COLORS, plot_mask

EXP_ROOT = Path(__file__).resolve().parent
SAM3_ROOT = EXP_ROOT.parent
RUNS_DIR = EXP_ROOT / "runs"
WEIGHTS_DIR = EXP_ROOT / "weights_modular"

BPE_PATH = SAM3_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
IMAGE_PATH = SAM3_ROOT / "assets" / "images" / "test_image.jpg"
PROMPT_TEXT = "shoe"
CONFIDENCE_THRESHOLD = 0.5
RESOLUTION = 1008

# 模块化 PT 说明 (基于 test_image + RESOLUTION=1008 的一次实测):
# - vision_backbone: 输入 [1, 3, 1008, 1008]
#   输出 backbone_fpn/vision_pos_enc 为 3 个尺度(高/中/低分辨率):
#   [1,256,288,288], [1,256,144,144], [1,256,72,72]
#   其中 vision_features 取的是最低分辨率一层 [1,256,72,72]
#   vision_pos_enc 由 PositionEmbeddingSine 在每个尺度特征图上计算(sin/cos)

# - text_encoder: 输入 1 条文本
#   输出 language_features [32,1,256], language_mask [1,32], language_embeds [32,1,1024]
#   language_features 为融合/匹配用的 256 维投影特征
#   language_mask 为 token 有效性掩码(避免 padding 参与注意力)
#   language_embeds 为更高维的原始文本嵌入(用于高维语义或兼容分支)
#   单词会先经过 BPE 分词并加特殊符号，因此会拆成多个 token
#   本例 prompt="shoe" 的 token 长度为 32

# - vl_backbone: 组合视觉+文本，输出为 backbone_out 字典(含上述字段)
#   其中 backbone_out 包含:
#   - backbone_fpn/vision_pos_enc: 3 个尺度的特征与位置编码
#   - vision_features: 最低分辨率一层(来自 3 个尺度中的最后一层)
#   - language_features/language_mask/language_embeds: 文本侧输出

# - transformer: 输入图像特征+文本特征+几何提示
#   输出 pred_boxes [1,200,4], pred_logits [1,200,1], pred_masks [1,200,288,288]

# - segmentation_head: 把查询特征解码为低分辨率 mask logits (同 pred_masks)

# - geometry_encoder: 将几何提示(点/框/掩码)编码为 prompt tokens，供 transformer 融合

# - dot_product_scoring: 计算查询与文本特征的匹配分数(对应 pred_logits)

# - transformer: 同时输出 queries [1,200,256] 与 presence_feats [1,1,256]
#   其中 queries 为 200 个查询向量，presence_feats 为文本存在性特征

# - postprocess: boxes [12,4], masks [12,1,720,1280], scores [12]
#   boxes 为像素坐标 xyxy，masks 上采样回原图尺寸


def visualize(image: Image.Image, masks, boxes, scores, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)

    if masks is not None and boxes is not None and len(masks) > 0:
        for idx in range(len(masks)):
            mask = masks[idx].detach().cpu().numpy()
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            color = COLORS[idx % len(COLORS)]
            plot_mask(mask, color=color, ax=ax)

            box = boxes[idx].detach().cpu().numpy()
            x0, y0, x1, y1 = box.tolist()
            rect = plt.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="lime",
                linewidth=2,
            )
            ax.add_patch(rect)
            if scores is not None and len(scores) > idx:
                ax.text(
                    x0,
                    y0,
                    f"{scores[idx].item():.3f}",
                    color="lime",
                    fontsize=10,
                    bbox=dict(facecolor="black", alpha=0.4, pad=2, edgecolor="none"),
                )

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_module_weights(module, name: str) -> None:
    path = WEIGHTS_DIR / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run run_text_det_modular.py to export module weights."
        )
    module.load_state_dict(torch.load(path, map_location="cpu"))


def build_model(device: str):
    # Construct each module explicitly so we can load per-module weights.
    vision_backbone = _create_vision_backbone(compile_mode=None, enable_inst_interactivity=True)
    text_encoder = _create_text_encoder(str(BPE_PATH))
    backbone = _create_vl_backbone(vision_backbone, text_encoder)
    transformer = _create_sam3_transformer()
    dot_prod_scoring = _create_dot_product_scoring()
    segmentation_head = _create_segmentation_head()
    input_geometry_encoder = _create_geometry_encoder()

    load_module_weights(vision_backbone, "vision_backbone")
    load_module_weights(text_encoder, "text_encoder")
    load_module_weights(transformer, "transformer")
    load_module_weights(segmentation_head, "segmentation_head")
    load_module_weights(input_geometry_encoder, "geometry_encoder")
    load_module_weights(dot_prod_scoring, "dot_product_scoring")

    model = _create_sam3_model(
        backbone=backbone,
        transformer=transformer,
        input_geometry_encoder=input_geometry_encoder,
        segmentation_head=segmentation_head,
        dot_prod_scoring=dot_prod_scoring,
        inst_interactive_predictor=None,
        eval_mode=True,
    )

    model.to(device=device)
    model.eval()
    return model


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    image = Image.open(IMAGE_PATH).convert("RGB")
    model = build_model(device)

    with torch.inference_mode():
        # Image preproc matches run_text_det_modular.py to keep outputs aligned.
        transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(RESOLUTION, RESOLUTION)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        image_t = v2.functional.to_image(image).to(device)
        image_t = transform(image_t).unsqueeze(0)
        img_w, img_h = image.size

        # find_stage 统一打包提示: 文本索引 + 几何提示(点/框/掩码)
        # 这里只使用文本, 几何提示为空
        find_stage = FindStage(
            img_ids=torch.tensor([0], device=device, dtype=torch.long),
            text_ids=torch.tensor([0], device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

        # Vision backbone outputs multi-scale features + positional encodings.
        backbone_out = model.backbone.forward_image(image_t)
        # Text encoder outputs token features for the prompt.
        text_outputs = model.backbone.forward_text([PROMPT_TEXT], device=device)
        backbone_out.update(text_outputs)

        # Geometry prompt is empty in this example (text-only).
        # If you add box/points prompts, use:
        # geometric_prompt = model._get_geo_prompt_from_find_input(find_stage)
        # (Note: with empty prompts this path errors, so keep dummy here.)
        geometric_prompt = model._get_dummy_prompt()

        # Detector forward (forward_grounding) 核心流程:
        # 1) _encode_prompt: 结合几何提示与文本特征生成 prompt tokens
        # 2) _run_encoder: 视觉/文本特征进入 encoder 得到 encoder_hidden_states
        # 3) _run_decoder: 解码得到 queries 与 pred_logits/pred_boxes
        # 4) _run_segmentation_heads: 生成低分辨率 pred_masks
        outputs = model.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_stage,
            geometric_prompt=geometric_prompt,
            find_target=None,
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]

        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > CONFIDENCE_THRESHOLD
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]

        # Convert normalized center-format boxes to pixel xyxy.
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h], device=device)
        boxes = boxes * scale_fct[None, :]

        # Upsample low-res masks to the original image size.
        out_masks = interpolate(
            out_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        masks = out_masks > 0.5
        scores = out_probs

    print("masks:", masks.shape)
    print("boxes:", boxes.shape)
    print("scores:", scores.shape)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / "text_det_vis_modular_from_pt.png"
    visualize(image, masks, boxes, scores, out_path)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
