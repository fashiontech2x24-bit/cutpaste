"""
Cut & Paste: Person Background Replacement using SAM3 + MiDaS.

Pipeline:
  1. SAM3  → person mask
  2. Mask erosion + Gaussian feather (removes edge fringing)
  3. MiDaS → background depth map
  4. Cover-crop background to 768×1024
  5. Depth-aware person scale + foot-anchored placement
  6. Alpha composite with edge spill suppression
  7. Depth-driven drop shadow below feet
  8. Optional Reinhard color harmonization
"""

import torch
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

_sam_model     = None
_sam_processor = None
_midas         = None
_midas_t       = None

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
    Erode binary mask inward then Gaussian-feather.
    Erosion discards fringe pixels mixed with the original shoot background
    before feathering, giving a cleaner blend boundary.
    """
    from scipy.ndimage import binary_erosion
    binary = mask > 0.5
    if erode_px > 0:
        binary = binary_erosion(binary, iterations=erode_px)
    return np.clip(gaussian_filter(binary.astype(np.float32), sigma=feather_sigma), 0.0, 1.0)


# ---------------------------------------------------------------------------
# MiDaS — monocular depth estimation (small, ~40ms, auto-downloads via hub)
# ---------------------------------------------------------------------------

def _load_midas(device="cuda"):
    global _midas, _midas_t
    if _midas is not None:
        return _midas, _midas_t
    print("Loading MiDaS small...")
    _midas   = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    _midas.to(device).eval()
    _midas_t = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
    print("MiDaS loaded.")
    return _midas, _midas_t


def _estimate_depth(image: Image.Image, device="cuda") -> np.ndarray:
    """Returns float32 depth map same size as image. 1.0 = closest, 0.0 = farthest."""
    model, transform = _load_midas(device)
    img_np = np.array(image)
    batch  = transform(img_np).to(device)
    with torch.no_grad():
        depth = model(batch)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=img_np.shape[:2],
            mode="bicubic", align_corners=False,
        ).squeeze()
    d = depth.cpu().numpy().astype(np.float32)
    return (d - d.min()) / (d.max() - d.min() + 1e-8)


def _drop_shadow(canvas, canvas_mask, foot_x, foot_y, person_w, ground_depth,
                 strength=0.55):
    """
    Elliptical drop shadow anchored at (foot_x, foot_y).
    Shadow is larger/darker when ground is close (high depth value).
    """
    sw      = int(person_w * 0.55)
    sh      = max(int(sw * 0.18 * (0.4 + ground_depth * 0.6)), 4)
    opacity = strength * (0.3 + ground_depth * 0.7)

    Y, X = np.ogrid[:OUTPUT_H, :OUTPUT_W]
    dist   = ((X - foot_x) / (sw + 1e-6)) ** 2 + ((Y - foot_y) / (sh + 1e-6)) ** 2
    shadow = np.clip(1.0 - dist, 0.0, 1.0)
    shadow = gaussian_filter(shadow, sigma=sh * 0.6)
    shadow = shadow / (shadow.max() + 1e-8)
    # Don't overdraw on solid person pixels
    shadow = shadow * (1.0 - np.clip(canvas_mask - 0.7, 0, 0.3) / 0.3)
    shadow = shadow[..., np.newaxis] * opacity

    out = canvas.copy()
    out = np.clip(out * (1.0 - shadow), 0, 255)
    return out


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


def _ambient_light_match(composite_arr, canvas_mask, strength=0.45):
    """
    Shift the person's Lab values toward the background scene.
    - a* / b*: full strength  → color temperature / hue match
    - L*     : half strength  → gentle brightness match so neutral/gray
                                backgrounds (zero chroma difference) still
                                produce a visible lighting adjustment.
    """
    img = composite_arr / 255.0
    lab = _rgb_to_lab(img)

    fg = canvas_mask > 0.5
    bg = canvas_mask < 0.1

    if fg.sum() < 100 or bg.sum() < 100:
        return composite_arr

    result_lab = lab.copy()
    strengths = [strength * 0.5, strength, strength]   # L*, a*, b*
    for ch in range(3):
        bg_mean = lab[..., ch][bg].mean()
        fg_mean = lab[..., ch][fg].mean()
        shift   = (bg_mean - fg_mean) * strengths[ch]
        result_lab[..., ch] = np.where(
            canvas_mask > 0.05,
            lab[..., ch] + shift * canvas_mask,
            lab[..., ch],
        )

    return np.clip(_lab_to_rgb(result_lab) * 255.0, 0, 255).astype(np.float32)


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
    person_fill=0.90,
    foot_anchor=0.90,
    shadow=True,
    shadow_strength=0.55,
    harmonize=False,
    ambient_light=True,
    ambient_strength=0.45,
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
        harmonize:            Run Reinhard color transfer (adjusts L, a*, b*).
        ambient_light:        Shift person color temperature to match scene (a*, b* only).
        ambient_strength:     Blend strength for ambient light shift (default 0.30).
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

    # --- MiDaS depth: sample ground depth at foot-landing point ---
    foot_y  = int(OUTPUT_H * foot_anchor)   # pixel row where feet touch ground
    foot_x  = OUTPUT_W // 2
    bg_depth = _estimate_depth(background, device=device)
    # Sample a small patch around the foot landing point for stability
    py0 = max(foot_y - 10, 0);  py1 = min(foot_y + 10, OUTPUT_H)
    px0 = max(foot_x - 30, 0);  px1 = min(foot_x + 30, OUTPUT_W)
    ground_depth = float(bg_depth[py0:py1, px0:px1].mean())  # 0=far, 1=close

    # --- Scale: height-driven only so person_fill is always honoured ---
    # The old min(width, height) let wide portraits be constrained by width,
    # making the person far shorter than person_fill requested.
    # We now scale by height exclusively; if the result is wider than the canvas
    # the sides are symmetrically clipped (person body stays centred).
    p_w, p_h  = portrait.size
    fit_scale = OUTPUT_H * person_fill / p_h
    new_w     = int(p_w * fit_scale)
    new_h     = int(p_h * fit_scale)

    portrait_resized = portrait.resize((new_w, new_h), Image.LANCZOS)
    mask_resized = np.array(
        Image.fromarray((mask * 255).astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)
    ).astype(np.float32) / 255.0

    # --- Foot-anchored placement ---
    # Centre horizontally; clip if new_w > OUTPUT_W (wide portraits).
    x_off = (OUTPUT_W - new_w) // 2        # negative when person wider than canvas
    y_off = max(foot_y - new_h, 0)         # clamp so head never goes above frame top

    # Canvas & person slice coordinates (handle wide-portrait clipping)
    xc0 = max(x_off, 0);          xc1 = min(x_off + new_w, OUTPUT_W)
    xp0 = xc0 - x_off;            xp1 = xp0 + (xc1 - xc0)
    yc0 = y_off;                   yc1 = min(y_off + new_h, OUTPUT_H)
    yp0 = 0;                       yp1 = yc1 - yc0

    # --- Canvas mask ---
    canvas_mask = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.float32)
    canvas_mask[yc0:yc1, xc0:xc1] = mask_resized[yp0:yp1, xp0:xp1]

    # --- Alpha composite with edge spill suppression ---
    result_arr  = np.array(background).astype(np.float32)
    person_arr  = np.array(portrait_resized).astype(np.float32)
    alpha       = mask_resized[yp0:yp1, xp0:xp1][..., np.newaxis]
    region      = result_arr[yc0:yc1, xc0:xc1]
    person_crop = person_arr[yp0:yp1, xp0:xp1]

    edge_zone    = 4.0 * alpha * (1.0 - alpha)
    SPILL        = 0.45
    person_clean = person_crop * (1.0 - edge_zone * SPILL) + region * (edge_zone * SPILL)
    result_arr[yc0:yc1, xc0:xc1] = person_clean * alpha + region * (1.0 - alpha)

    # --- Find actual foot position from mask bottom (not the target anchor) ---
    # foot_anchor is where we AIM to place the feet, but the portrait may have
    # blank space below the person inside its bounding box.  Using foot_y directly
    # as the shadow center leaves a visible gap between the person and the shadow.
    mask_rows = np.any(canvas_mask > 0.1, axis=1)
    actual_foot_y = int(np.where(mask_rows)[0][-1]) if mask_rows.any() else foot_y

    # --- Depth-driven drop shadow anchored to actual feet ---
    if shadow:
        result_arr = _drop_shadow(result_arr, canvas_mask, foot_x, actual_foot_y,
                                  new_w, ground_depth, strength=shadow_strength)

    # --- Ambient light: color-temperature match (chroma only, no luminance change) ---
    if ambient_light:
        result_arr = _ambient_light_match(result_arr, canvas_mask,
                                          strength=ambient_strength)

    # --- Reinhard color harmonization (full Lab: tone + color) ---
    if harmonize:
        result_arr = _reinhard_transfer(result_arr, canvas_mask)

    Image.fromarray(result_arr.astype(np.uint8)).save(output_path)
    label = "harmonized" if harmonize else "composited"
    print(f"Saved {OUTPUT_W}×{OUTPUT_H} {label} image → {output_path}")
    return output_path
