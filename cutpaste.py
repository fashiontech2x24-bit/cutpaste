"""
Cut & Paste: Person background replacement using SAM3.

Pipeline:
  1. SAM3  → person mask
  2. Mask erosion + Gaussian feather (clean edges)
  3. Cover-crop background to 768×1024
  4. Scale person to person_fill of frame height (by mask bbox)
  5. Foot-anchored placement
  6. Alpha composite with edge spill suppression
"""

import torch
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

_sam_model     = None
_sam_processor = None

MODEL_ID = "facebook/sam3"
OUTPUT_W, OUTPUT_H = 768, 1024


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


def _segment_person(image, prompt="person", confidence_threshold=0.5, device="cuda"):
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

    masks, scores = results["masks"], results["scores"]
    if len(masks) == 0:
        return None

    best = scores.argmax().item()
    return masks[best].cpu().numpy().astype(np.float32)


def _refine_mask(mask, erode_px=2, feather_sigma=1.5):
    from scipy.ndimage import binary_erosion
    binary = mask > 0.5
    if erode_px > 0:
        binary = binary_erosion(binary, iterations=erode_px)
    return np.clip(gaussian_filter(binary.astype(np.float32), sigma=feather_sigma), 0.0, 1.0)


def replace_background(
    portrait_path,
    background_path,
    output_path,
    prompt="person",
    confidence_threshold=0.5,
    erode_px=2,
    feather_sigma=1.5,
    person_fill=0.92,
    foot_anchor=0.92,
    device="cuda",
):
    portrait   = Image.open(portrait_path).convert("RGB")
    background = Image.open(background_path).convert("RGB")

    # Cover-crop background to OUTPUT_W × OUTPUT_H
    bg_w, bg_h = background.size
    cover      = max(OUTPUT_W / bg_w, OUTPUT_H / bg_h)
    bg_sw, bg_sh = int(bg_w * cover), int(bg_h * cover)
    background = background.resize((bg_sw, bg_sh), Image.LANCZOS)
    cx, cy     = (bg_sw - OUTPUT_W) // 2, (bg_sh - OUTPUT_H) // 2
    background = background.crop((cx, cy, cx + OUTPUT_W, cy + OUTPUT_H))

    # Segment
    mask = _segment_person(portrait, prompt=prompt,
                           confidence_threshold=confidence_threshold, device=device)
    if mask is None:
        raise ValueError(f"No '{prompt}' detected above threshold {confidence_threshold}")

    mask = _refine_mask(mask, erode_px=erode_px, feather_sigma=feather_sigma)

    # Scale by mask bounding box so person_fill = actual body fraction of frame
    p_w, p_h = portrait.size
    mask_rows = np.any(mask > 0.5, axis=1)
    if mask_rows.any():
        mb_top = int(np.where(mask_rows)[0][0])
        mb_bot = int(np.where(mask_rows)[0][-1])
    else:
        mb_top, mb_bot = 0, p_h - 1
    person_body_h = mb_bot - mb_top + 1

    fit_scale = OUTPUT_H * person_fill / person_body_h
    new_w     = int(p_w * fit_scale)
    new_h     = int(p_h * fit_scale)

    portrait_resized = portrait.resize((new_w, new_h), Image.LANCZOS)
    mask_resized = np.array(
        Image.fromarray((mask * 255).astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)
    ).astype(np.float32) / 255.0

    mb_bot_scaled = int(mb_bot * fit_scale)
    foot_y = int(OUTPUT_H * foot_anchor)
    x_off  = (OUTPUT_W - new_w) // 2
    y_off  = max(foot_y - mb_bot_scaled, 0)

    # Canvas & person slice coords (clips wide portraits at canvas edges)
    xc0 = max(x_off, 0);      xc1 = min(x_off + new_w, OUTPUT_W)
    xp0 = xc0 - x_off;        xp1 = xp0 + (xc1 - xc0)
    yc0 = y_off;               yc1 = min(y_off + new_h, OUTPUT_H)
    yp0 = 0;                   yp1 = yc1 - yc0

    # Alpha composite with edge spill suppression
    result_arr  = np.array(background).astype(np.float32)
    person_arr  = np.array(portrait_resized).astype(np.float32)
    alpha       = mask_resized[yp0:yp1, xp0:xp1][..., np.newaxis]
    region      = result_arr[yc0:yc1, xc0:xc1]
    person_crop = person_arr[yp0:yp1, xp0:xp1]

    edge_zone    = 4.0 * alpha * (1.0 - alpha)
    SPILL        = 0.10
    person_clean = person_crop * (1.0 - edge_zone * SPILL) + region * (edge_zone * SPILL)
    result_arr[yc0:yc1, xc0:xc1] = person_clean * alpha + region * (1.0 - alpha)

    Image.fromarray(result_arr.astype(np.uint8)).save(output_path)
    print(f"Saved {OUTPUT_W}×{OUTPUT_H} composite → {output_path}")
    return output_path
