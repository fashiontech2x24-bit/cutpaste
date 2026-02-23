"""
Cut & Paste: Person Background Replacement using SAM3.

Segments a person from a portrait image using SAM3 (Segment Anything Model 3)
and composites them onto a new background image.
"""

import torch
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# Lazy-loaded globals for model caching
_model = None
_processor = None

MODEL_ID = "facebook/sam3"


def _load_model(device="cuda"):
    """Load SAM3 model and processor (singleton, cached after first call)."""
    global _model, _processor

    if _model is not None:
        return _model, _processor

    from transformers import Sam3Model, Sam3Processor

    print(f"Loading SAM3 model from {MODEL_ID}...")
    _processor = Sam3Processor.from_pretrained(MODEL_ID)
    _model = Sam3Model.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(device)
    _model.eval()
    print("SAM3 model loaded.")

    return _model, _processor


def _segment_person(image, prompt="person", confidence_threshold=0.5, device="cuda"):
    """
    Run SAM3 text-prompted segmentation and return the best mask.

    Returns:
        numpy array (H, W) float32 mask with values 0.0-1.0, or None if no match.
    """
    model, processor = _load_model(device)

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

    # Pick the highest-scoring mask
    best_idx = scores.argmax().item()
    mask = masks[best_idx].cpu().numpy().astype(np.float32)

    return mask


def _feather_mask(mask, sigma=3.0):
    """Apply Gaussian feathering to mask edges for smooth compositing."""
    feathered = gaussian_filter(mask, sigma=sigma)
    # Keep the solid interior, only soften the edges
    feathered = np.clip(feathered, 0.0, 1.0)
    return feathered


def replace_background(
    portrait_path,
    background_path,
    output_path,
    prompt="person",
    confidence_threshold=0.5,
    feather_sigma=3.0,
    device="cuda",
):
    """
    Segments a person from portrait_path using SAM3 and composites onto background_path.

    Args:
        portrait_path: Path to the portrait/source image.
        background_path: Path to the new background image.
        output_path: Path to save the composited result.
        prompt: Text prompt for segmentation (default: "person").
        confidence_threshold: Minimum confidence for mask selection.
        feather_sigma: Gaussian blur sigma for edge feathering.
        device: Torch device ("cuda" or "cpu").

    Returns:
        Path to the saved output image.
    """
    portrait = Image.open(portrait_path).convert("RGB")
    background = Image.open(background_path).convert("RGB")

    # Segment the person
    mask = _segment_person(portrait, prompt=prompt, confidence_threshold=confidence_threshold, device=device)

    if mask is None:
        raise ValueError(f"No '{prompt}' detected in {portrait_path} above threshold {confidence_threshold}")

    # Feather the mask edges
    mask = _feather_mask(mask, sigma=feather_sigma)

    # Resize person + mask to fit on background, maintaining aspect ratio
    bg_w, bg_h = background.size
    p_w, p_h = portrait.size

    scale = min(bg_w / p_w, bg_h / p_h)
    new_w = int(p_w * scale)
    new_h = int(p_h * scale)

    portrait_resized = portrait.resize((new_w, new_h), Image.LANCZOS)
    mask_resized = np.array(
        Image.fromarray((mask * 255).astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)
    ).astype(np.float32) / 255.0

    # Center the person on the background
    x_offset = (bg_w - new_w) // 2
    y_offset = (bg_h - new_h) // 2

    # Composite
    result = background.copy()
    portrait_arr = np.array(portrait_resized).astype(np.float32)
    result_arr = np.array(result).astype(np.float32)

    # Apply mask as alpha blend in the placement region
    region = result_arr[y_offset : y_offset + new_h, x_offset : x_offset + new_w]
    alpha = mask_resized[..., np.newaxis]  # (H, W, 1)
    blended = portrait_arr * alpha + region * (1.0 - alpha)
    result_arr[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = blended

    result = Image.fromarray(result_arr.astype(np.uint8))
    result.save(output_path)
    print(f"Saved composited image to {output_path}")

    return output_path
