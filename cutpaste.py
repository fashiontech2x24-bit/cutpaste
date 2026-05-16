"""
Cut & Paste: Person Background Replacement using SAM3.

Pipeline:
  1. SAM3 text-prompted segmentation → person mask
  2. Gaussian edge feathering
  3. Scale background to 768×1024 (cover-crop)
  4. Scale person to person_fill % of frame height, alpha-composite
  5. Reinhard color transfer — matches person's L*a*b* tone/lighting to background
"""

import torch
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

_sam_model = None
_sam_processor = None

MODEL_ID = "facebook/sam3"
OUTPUT_W, OUTPUT_H = 768, 1024


# ---------------------------------------------------------------------------
# SAM3
# ---------------------------------------------------------------------------

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


def _refine_mask(mask, erode_px=4, feather_sigma=3.0):
    """
    1. Erode the binary mask inward by erode_px pixels.
       This pulls the boundary past the fringe zone where edge pixels
       contain a mix of person + original background color.
    2. Gaussian feather from the clean eroded boundary for smooth blending.

    Better than naive Gaussian-on-raw-mask which blends in fringe pixels.
    """
    from scipy.ndimage import binary_erosion

    binary = mask > 0.5
    if erode_px > 0:
        binary = binary_erosion(binary, iterations=erode_px)
    return np.clip(gaussian_filter(binary.astype(np.float32), sigma=feather_sigma), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Reinhard color transfer
# ---------------------------------------------------------------------------
# Transfers the color statistics (mean + std per L*a*b* channel) from the
# background region into the foreground (person), so their tone and lighting
# appear consistent.  Pure numpy — no extra dependencies.

def _rgb_to_lab(img_f32):
    """float32 RGB [0,1] → L*a*b* (OpenCV-style, float32)."""
    # sRGB → linear
    mask = img_f32 > 0.04045
    linear = np.where(mask, ((img_f32 + 0.055) / 1.055) ** 2.4, img_f32 / 12.92)
    # linear RGB → XYZ (D65)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    xyz = linear @ M.T
    # XYZ → L*a*b*
    xyz /= np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    eps = 0.008856
    xyz = np.where(xyz > eps, np.cbrt(xyz), (7.787 * xyz) + (16 / 116))
    L = 116 * xyz[..., 1] - 16
    a = 500 * (xyz[..., 0] - xyz[..., 1])
    b = 200 * (xyz[..., 1] - xyz[..., 2])
    return np.stack([L, a, b], axis=-1)


def _lab_to_rgb(lab):
    """L*a*b* float32 → float32 RGB [0,1]."""
    fy = (lab[..., 0] + 16) / 116
    fx = lab[..., 1] / 500 + fy
    fz = fy - lab[..., 2] / 200
    eps = 0.008856
    xyz = np.stack([
        np.where(fx ** 3 > eps, fx ** 3, (fx - 16 / 116) / 7.787),
        np.where(fy ** 3 > eps, fy ** 3, (fy - 16 / 116) / 7.787),
        np.where(fz ** 3 > eps, fz ** 3, (fz - 16 / 116) / 7.787),
    ], axis=-1)
    xyz *= np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    M_inv = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                      [-0.9692660,  1.8760108,  0.0415560],
                      [ 0.0556434, -0.2040259,  1.0572252]], dtype=np.float32)
    linear = xyz @ M_inv.T
    linear = np.clip(linear, 0.0, None)
    rgb = np.where(linear > 0.0031308,
                   1.055 * np.power(linear, 1 / 2.4) - 0.055,
                   12.92 * linear)
    return np.clip(rgb, 0.0, 1.0)


def _reinhard_transfer(composite_arr, canvas_mask, strength=0.35):
    """
    Gently shift the person's L*a*b* mean toward the background's mean.

    composite_arr: float32 RGB [0,255] — full composited canvas
    canvas_mask:   float32 [0,1]       — person mask on canvas
    strength:      0.0 = no effect, 1.0 = full transfer (default 0.35)

    Only the mean is transferred (no std scaling) to prevent color collapse
    when the background is near-uniform.
    """
    img = composite_arr / 255.0
    lab = _rgb_to_lab(img)

    fg = canvas_mask > 0.5   # solid person pixels
    bg = canvas_mask < 0.1   # solid background pixels

    if fg.sum() < 100 or bg.sum() < 100:
        return composite_arr  # not enough pixels to sample — skip

    result_lab = lab.copy()
    for ch in range(3):
        src_vals = lab[..., ch][fg]
        tgt_vals = lab[..., ch][bg]

        # Only shift the mean — do NOT scale by std ratio.
        # Scaling by tgt.std/src.std washes out the person when
        # the background is near-uniform (tgt.std ≈ 0).
        mean_shift = tgt_vals.mean() - src_vals.mean()

        # Blend: don't fully drag person to background, just nudge it
        result_lab[..., ch][fg] = src_vals + mean_shift * strength

    result_rgb = _lab_to_rgb(result_lab)
    return np.clip(result_rgb * 255.0, 0, 255).astype(np.float32)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def replace_background(
    portrait_path,
    background_path,
    output_path,
    prompt="person",
    confidence_threshold=0.5,
    erode_px=4,
    feather_sigma=3.0,
    person_fill=0.75,
    harmonize=False,
    device="cuda",
):
    """
    Segment person from portrait_path, composite onto background_path,
    then optionally harmonize colors via Reinhard Lab transfer.

    Args:
        portrait_path:        Source portrait image path.
        background_path:      New background image path.
        output_path:          Where to save the result PNG.
        prompt:               SAM3 text prompt (default: "person").
        confidence_threshold: Min SAM3 confidence for mask selection.
        feather_sigma:        Gaussian blur sigma for edge feathering.
        person_fill:          Person height as fraction of OUTPUT_H (default 0.75).
        harmonize:            Run Reinhard color transfer after compositing.
        device:               Torch device ("cuda" or "cpu").
    """
    portrait   = Image.open(portrait_path).convert("RGB")
    background = Image.open(background_path).convert("RGB")

    # Background → cover-crop to OUTPUT_W × OUTPUT_H
    bg_w, bg_h   = background.size
    cover        = max(OUTPUT_W / bg_w, OUTPUT_H / bg_h)
    bg_sw, bg_sh = int(bg_w * cover), int(bg_h * cover)
    background   = background.resize((bg_sw, bg_sh), Image.LANCZOS)
    cx, cy       = (bg_sw - OUTPUT_W) // 2, (bg_sh - OUTPUT_H) // 2
    background   = background.crop((cx, cy, cx + OUTPUT_W, cy + OUTPUT_H))

    # Segment
    mask = _segment_person(portrait, prompt=prompt,
                           confidence_threshold=confidence_threshold, device=device)
    if mask is None:
        raise ValueError(f"No '{prompt}' detected above threshold {confidence_threshold}")

    mask = _refine_mask(mask, erode_px=erode_px, feather_sigma=feather_sigma)

    # Scale person: height = person_fill × OUTPUT_H, width clamped to OUTPUT_W
    p_w, p_h  = portrait.size
    fit_scale = min(OUTPUT_W / p_w, OUTPUT_H * person_fill / p_h)
    new_w     = int(p_w * fit_scale)
    new_h     = int(p_h * fit_scale)

    portrait_resized = portrait.resize((new_w, new_h), Image.LANCZOS)
    mask_resized = np.array(
        Image.fromarray((mask * 255).astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)
    ).astype(np.float32) / 255.0

    # Center on canvas
    x_off = (OUTPUT_W - new_w) // 2
    y_off = (OUTPUT_H - new_h) // 2

    # Build canvas-sized mask for harmonization
    canvas_mask = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.float32)
    canvas_mask[y_off : y_off + new_h, x_off : x_off + new_w] = mask_resized

    # Alpha composite with edge spill suppression
    # Edge pixels of the portrait contain a mix of person + original background.
    # In the blend zone (0 < alpha < 1) we nudge person colors toward the new
    # background, which removes color fringing from the original shoot.
    result_arr   = np.array(background).astype(np.float32)
    person_arr   = np.array(portrait_resized).astype(np.float32)
    alpha        = mask_resized[..., np.newaxis]
    region       = result_arr[y_off : y_off + new_h, x_off : x_off + new_w]

    # Peaks at 1.0 where alpha=0.5 (the boundary), zero at solid interior/exterior
    edge_zone    = 4.0 * alpha * (1.0 - alpha)
    SPILL        = 0.45  # suppression strength — higher = cleaner edges, less person detail
    person_clean = person_arr * (1.0 - edge_zone * SPILL) + region * (edge_zone * SPILL)

    result_arr[y_off : y_off + new_h, x_off : x_off + new_w] = (
        person_clean * alpha + region * (1.0 - alpha)
    )

    # Reinhard color harmonization
    if harmonize:
        result_arr = _reinhard_transfer(result_arr, canvas_mask)

    Image.fromarray(result_arr.astype(np.uint8)).save(output_path)
    label = "harmonized" if harmonize else "composited"
    print(f"Saved {OUTPUT_W}×{OUTPUT_H} {label} image → {output_path}")
    return output_path
