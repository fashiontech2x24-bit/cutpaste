"""
Cut & Paste: Person Background Replacement using SAM3 + PCTNet harmonization.

Pipeline:
  1. SAM3 text-prompted segmentation → person mask
  2. Gaussian edge feathering
  3. Scale background to 768×1024 (cover-crop)
  4. Scale person to person_fill % of frame height, alpha-composite
  5. PCTNet harmonization → adjusts person colors/lighting to match background
"""

import torch
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# Cached model globals
_sam_model = None
_sam_processor = None
_harm_model = None

MODEL_ID = "facebook/sam3"

OUTPUT_W, OUTPUT_H = 768, 1024  # standard output canvas (portrait 3:4)


def _load_sam(device="cuda"):
    global _sam_model, _sam_processor
    if _sam_model is not None:
        return _sam_model, _sam_processor

    from transformers import Sam3Model, Sam3Processor

    print(f"Loading SAM3 from {MODEL_ID}...")
    _sam_processor = Sam3Processor.from_pretrained(MODEL_ID)
    _sam_model = Sam3Model.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(device)
    _sam_model.eval()
    print("SAM3 loaded.")
    return _sam_model, _sam_processor


def _load_harmonizer(device="cuda"):
    global _harm_model
    if _harm_model is not None:
        return _harm_model

    from libcom import ImageHarmonizationModel

    gpu_index = 0 if device == "cuda" else "cpu"
    print("Loading PCTNet harmonization model...")
    _harm_model = ImageHarmonizationModel(device=gpu_index, model_type="PCTNet")
    print("PCTNet loaded.")
    return _harm_model


def _segment_person(image, prompt="person", confidence_threshold=0.5, device="cuda"):
    """Run SAM3 text-prompted segmentation; returns best float32 mask or None."""
    model, processor = _load_sam(device)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=confidence_threshold,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    masks = results["masks"]
    scores = results["scores"]

    if len(masks) == 0:
        return None

    best_idx = scores.argmax().item()
    return masks[best_idx].cpu().numpy().astype(np.float32)


def _feather_mask(mask, sigma=3.0):
    """Gaussian feathering on mask edges for smooth compositing."""
    return np.clip(gaussian_filter(mask, sigma=sigma), 0.0, 1.0)


def replace_background(
    portrait_path,
    background_path,
    output_path,
    prompt="person",
    confidence_threshold=0.5,
    feather_sigma=3.0,
    person_fill=0.75,
    harmonize=True,
    device="cuda",
):
    """
    Segment person from portrait_path, composite onto background_path,
    then optionally harmonize colors/lighting with PCTNet.

    Args:
        portrait_path:        Source portrait image path.
        background_path:      New background image path.
        output_path:          Where to save the result PNG.
        prompt:               SAM3 text prompt (default: "person").
        confidence_threshold: Min SAM3 confidence for mask selection.
        feather_sigma:        Gaussian blur sigma for edge feathering.
        person_fill:          Person height as fraction of OUTPUT_H (default 0.75 = 75%).
        harmonize:            Run PCTNet harmonization after compositing (default True).
        device:               Torch device string ("cuda" or "cpu").

    Returns:
        Path to the saved output image.
    """
    portrait = Image.open(portrait_path).convert("RGB")
    background = Image.open(background_path).convert("RGB")

    # --- Background: scale-to-cover then center-crop to OUTPUT_W × OUTPUT_H ---
    bg_w, bg_h = background.size
    cover_scale = max(OUTPUT_W / bg_w, OUTPUT_H / bg_h)
    bg_sw = int(bg_w * cover_scale)
    bg_sh = int(bg_h * cover_scale)
    background = background.resize((bg_sw, bg_sh), Image.LANCZOS)
    cx = (bg_sw - OUTPUT_W) // 2
    cy = (bg_sh - OUTPUT_H) // 2
    background = background.crop((cx, cy, cx + OUTPUT_W, cy + OUTPUT_H))

    # --- Segmentation ---
    mask = _segment_person(portrait, prompt=prompt, confidence_threshold=confidence_threshold, device=device)
    if mask is None:
        raise ValueError(f"No '{prompt}' detected in {portrait_path} above threshold {confidence_threshold}")

    mask = _feather_mask(mask, sigma=feather_sigma)

    # --- Scale person: height = person_fill × OUTPUT_H, width clamped to OUTPUT_W ---
    p_w, p_h = portrait.size
    fit_scale = min(OUTPUT_W / p_w, OUTPUT_H * person_fill / p_h)
    new_w = int(p_w * fit_scale)
    new_h = int(p_h * fit_scale)

    portrait_resized = portrait.resize((new_w, new_h), Image.LANCZOS)
    mask_resized = np.array(
        Image.fromarray((mask * 255).astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)
    ).astype(np.float32) / 255.0

    # --- Center person on canvas ---
    x_off = (OUTPUT_W - new_w) // 2
    y_off = (OUTPUT_H - new_h) // 2

    # --- Alpha composite ---
    result_arr = np.array(background).astype(np.float32)
    portrait_arr = np.array(portrait_resized).astype(np.float32)
    alpha = mask_resized[..., np.newaxis]
    region = result_arr[y_off : y_off + new_h, x_off : x_off + new_w]
    result_arr[y_off : y_off + new_h, x_off : x_off + new_w] = (
        portrait_arr * alpha + region * (1.0 - alpha)
    )

    # --- PCTNet harmonization ---
    # Adjusts the person's colors and lighting to match the background scene.
    # libcom expects BGR uint8 composite + uint8 mask (255 = foreground).
    if harmonize:
        harmonizer = _load_harmonizer(device)

        composite_bgr = result_arr.astype(np.uint8)[..., ::-1].copy()

        canvas_mask = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.uint8)
        canvas_mask[y_off : y_off + new_h, x_off : x_off + new_w] = (
            (mask_resized * 255).astype(np.uint8)
        )

        harmonized_bgr = harmonizer(composite_bgr, canvas_mask)
        result_arr = harmonized_bgr[..., ::-1].astype(np.float32)  # back to RGB

    result = Image.fromarray(result_arr.astype(np.uint8))
    result.save(output_path)
    label = "harmonized" if harmonize else "composited"
    print(f"Saved {OUTPUT_W}×{OUTPUT_H} {label} image to {output_path}")

    return output_path
