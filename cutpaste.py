"""
Cut & Paste — rembg + MiDaS pipeline.

  rembg (u2net_human_seg) : portrait alpha matting (~30ms, auto-downloads ~170MB)
  MiDaS small             : depth estimation       (~40ms, auto-downloads ~80MB)

Pipeline:
  1. rembg   → clean alpha matte (hair/edge aware, GPU ONNX)
  2. MiDaS   → depth map of background
  3. Cover-crop background to 768×1024
  4. Scale person to person_fill % of frame height
  5. Alpha composite
  6. Depth-driven drop shadow below feet
  7. Optional Reinhard color harmonization
"""

import torch
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

OUTPUT_W, OUTPUT_H = 768, 1024

_rembg_session = None
_midas  = None
_midas_t = None


# ---------------------------------------------------------------------------
# rembg — portrait alpha matting (u2net_human_seg, GPU ONNX)
# ---------------------------------------------------------------------------

def _load_rembg():
    global _rembg_session
    if _rembg_session is not None:
        return _rembg_session
    from rembg import new_session
    print("Loading rembg u2net_human_seg...")
    _rembg_session = new_session("u2net_human_seg")
    print("rembg ready.")
    return _rembg_session


def _matte_person(image: Image.Image, device="cuda") -> np.ndarray:
    """
    rembg inference on a PIL RGB image.
    Returns float32 alpha matte, same size as input, values 0.0-1.0.
    """
    from rembg import remove
    session = _load_rembg()
    rgba    = remove(image, session=session)          # PIL RGBA
    alpha   = np.array(rgba)[:, :, 3].astype(np.float32) / 255.0
    return alpha


# ---------------------------------------------------------------------------
# MiDaS — monocular depth estimation (small variant, ~40ms)
# ---------------------------------------------------------------------------

def _load_midas(device="cuda"):
    global _midas, _midas_t
    if _midas is not None:
        return _midas, _midas_t

    print("Loading MiDaS small...")
    _midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    _midas.to(device).eval()
    _midas_t = torch.hub.load("intel-isl/MiDaS", "transforms",
                               trust_repo=True).small_transform
    print("MiDaS loaded.")
    return _midas, _midas_t


def _estimate_depth(image: Image.Image, device="cuda") -> np.ndarray:
    """
    MiDaS depth map, resized to image dimensions.
    Returns float32 [0,1] where 1.0 = closest to camera.
    """
    model, transform = _load_midas(device)
    img_np = np.array(image)
    batch  = transform(img_np).to(device)

    with torch.no_grad():
        depth = model(batch)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=img_np.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    d = depth.cpu().numpy().astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return d


# ---------------------------------------------------------------------------
# Shadow generation (depth-aware)
# ---------------------------------------------------------------------------

def _add_shadow(result_arr, canvas_mask, bg_depth, x_off, y_off,
                new_w, new_h, shadow_strength=0.55):
    """
    Cast an elliptical drop shadow below the person's feet.
    Shadow opacity is scaled by the background depth at that point —
    darker/larger when ground is close, lighter/smaller when far away.
    """
    # Foot position = bottom center of person bounding box
    foot_y = min(y_off + new_h, OUTPUT_H - 1)
    foot_x = x_off + new_w // 2

    # Sample background depth at foot position (clamp to valid range)
    sample_y = min(foot_y + 5, OUTPUT_H - 1)
    ground_depth = bg_depth[sample_y, foot_x]  # 0=far, 1=close

    # Shadow ellipse dimensions — scale with person width and depth
    sw = int(new_w * 0.55)
    sh = max(int(sw * 0.18 * (0.4 + ground_depth * 0.6)), 4)
    opacity = shadow_strength * (0.3 + ground_depth * 0.7)

    # Build shadow mask (smooth ellipse)
    cy, cx = foot_y, foot_x
    Y, X = np.ogrid[:OUTPUT_H, :OUTPUT_W]
    dist = ((X - cx) / (sw + 1e-6)) ** 2 + ((Y - cy) / (sh + 1e-6)) ** 2
    shadow = np.clip(1.0 - dist, 0.0, 1.0)
    shadow = gaussian_filter(shadow, sigma=sh * 0.6)
    shadow = shadow / (shadow.max() + 1e-8)

    # Don't draw shadow behind the solid person (mask > 0.7)
    shadow = shadow * (1.0 - np.clip(canvas_mask - 0.7, 0, 0.3) / 0.3)
    shadow = shadow[..., np.newaxis] * opacity

    result_arr = result_arr.copy()
    result_arr = result_arr * (1.0 - shadow)
    return result_arr


# ---------------------------------------------------------------------------
# Reinhard Lab color harmonization (mean-shift only, safe for flat BGs)
# ---------------------------------------------------------------------------

def _rgb_to_lab(img_f32):
    mask   = img_f32 > 0.04045
    linear = np.where(mask, ((img_f32 + 0.055) / 1.055) ** 2.4, img_f32 / 12.92)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    xyz = linear @ M.T
    xyz /= np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    eps = 0.008856
    xyz = np.where(xyz > eps, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * xyz[..., 1] - 16
    a = 500 * (xyz[..., 0] - xyz[..., 1])
    b = 200 * (xyz[..., 1] - xyz[..., 2])
    return np.stack([L, a, b], axis=-1)


def _lab_to_rgb(lab):
    fy = (lab[..., 0] + 16) / 116
    fx = lab[..., 1] / 500 + fy
    fz = fy - lab[..., 2] / 200
    eps = 0.008856
    xyz = np.stack([
        np.where(fx**3 > eps, fx**3, (fx - 16/116) / 7.787),
        np.where(fy**3 > eps, fy**3, (fy - 16/116) / 7.787),
        np.where(fz**3 > eps, fz**3, (fz - 16/116) / 7.787),
    ], axis=-1)
    xyz *= np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    M_inv = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                      [-0.9692660,  1.8760108,  0.0415560],
                      [ 0.0556434, -0.2040259,  1.0572252]], dtype=np.float32)
    linear = np.clip(xyz @ M_inv.T, 0.0, None)
    rgb = np.where(linear > 0.0031308,
                   1.055 * np.power(linear, 1/2.4) - 0.055,
                   12.92 * linear)
    return np.clip(rgb, 0.0, 1.0)


def _reinhard_transfer(composite_arr, canvas_mask, strength=0.35):
    img = composite_arr / 255.0
    lab = _rgb_to_lab(img)
    fg  = canvas_mask > 0.5
    bg  = canvas_mask < 0.1
    if fg.sum() < 100 or bg.sum() < 100:
        return composite_arr
    result_lab = lab.copy()
    for ch in range(3):
        src = lab[..., ch][fg]
        tgt = lab[..., ch][bg]
        result_lab[..., ch][fg] = src + (tgt.mean() - src.mean()) * strength
    return np.clip(_lab_to_rgb(result_lab) * 255.0, 0, 255).astype(np.float32)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def replace_background(
    portrait_path,
    background_path,
    output_path,
    person_fill=0.75,
    harmonize=False,
    shadow=True,
    shadow_strength=0.55,
    device="cuda",
):
    """
    Composite a person onto a new background using MODNet + MiDaS.

    Args:
        portrait_path:    Source portrait image path.
        background_path:  New background image path.
        output_path:      Where to save the 768×1024 result PNG.
        person_fill:      Person height as fraction of OUTPUT_H (default 0.75).
        harmonize:        Apply Reinhard color shift (default False).
        shadow:           Render depth-aware drop shadow (default True).
        shadow_strength:  Shadow opacity 0-1 (default 0.55).
        device:           Torch device string ("cuda" or "cpu").
    """
    portrait   = Image.open(portrait_path).convert("RGB")
    background = Image.open(background_path).convert("RGB")

    # --- MODNet: alpha matte ---
    matte = _matte_person(portrait, device=device)  # float32 [0,1], portrait size

    # --- MiDaS: background depth map ---
    if shadow:
        bg_depth = _estimate_depth(background, device=device)  # float32 [0,1]
    else:
        bg_depth = None

    # --- Background: cover-crop to 768×1024 ---
    bg_w, bg_h = background.size
    cover      = max(OUTPUT_W / bg_w, OUTPUT_H / bg_h)
    bg_sw      = int(bg_w * cover)
    bg_sh      = int(bg_h * cover)
    background = background.resize((bg_sw, bg_sh), Image.LANCZOS)
    cx, cy     = (bg_sw - OUTPUT_W) // 2, (bg_sh - OUTPUT_H) // 2
    background = background.crop((cx, cy, cx + OUTPUT_W, cy + OUTPUT_H))
    if bg_depth is not None:
        bg_depth_pil = Image.fromarray((bg_depth * 255).astype(np.uint8)).resize(
            (bg_sw, bg_sh), Image.BILINEAR
        ).crop((cx, cy, cx + OUTPUT_W, cy + OUTPUT_H))
        bg_depth = np.array(bg_depth_pil).astype(np.float32) / 255.0

    # --- Scale person: height = person_fill × OUTPUT_H ---
    p_w, p_h  = portrait.size
    fit_scale = min(OUTPUT_W / p_w, OUTPUT_H * person_fill / p_h)
    new_w     = int(p_w * fit_scale)
    new_h     = int(p_h * fit_scale)

    portrait_r = portrait.resize((new_w, new_h), Image.LANCZOS)
    matte_r    = np.array(
        Image.fromarray((matte * 255).astype(np.uint8)).resize(
            (new_w, new_h), Image.LANCZOS
        )
    ).astype(np.float32) / 255.0

    # --- Center on canvas ---
    x_off = (OUTPUT_W - new_w) // 2
    y_off = (OUTPUT_H - new_h) // 2

    # --- Canvas mask (full 768×1024) ---
    canvas_mask = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.float32)
    canvas_mask[y_off : y_off + new_h, x_off : x_off + new_w] = matte_r

    # --- Alpha composite ---
    result     = np.array(background).astype(np.float32)
    person_arr = np.array(portrait_r).astype(np.float32)
    alpha      = matte_r[..., np.newaxis]
    region     = result[y_off : y_off + new_h, x_off : x_off + new_w]
    result[y_off : y_off + new_h, x_off : x_off + new_w] = (
        person_arr * alpha + region * (1.0 - alpha)
    )

    # --- Depth-aware shadow ---
    if shadow and bg_depth is not None:
        result = _add_shadow(result, canvas_mask, bg_depth,
                             x_off, y_off, new_w, new_h, shadow_strength)

    # --- Color harmonization ---
    if harmonize:
        result = _reinhard_transfer(result, canvas_mask)

    Image.fromarray(result.astype(np.uint8)).save(output_path)
    print(f"Saved {OUTPUT_W}×{OUTPUT_H} → {output_path}")
    return output_path
