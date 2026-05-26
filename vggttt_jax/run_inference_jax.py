#!/usr/bin/env python
"""VGG-T³ JAX Inference Runner

Complete end-to-end inference pipeline that:
  1. Loads images from disk (PNG/JPG).
  2. Preprocesses them (resize, normalize) – mirrors the PyTorch pipeline.
  3. Loads the converted Flax msgpack weights.
  4. Runs the full VGGT forward pass (Aggregator → CameraHead → DepthHead → PointHead).
  5. Saves outputs to a NumPy .npz file.

Works on **both CPU and GPU** – select via --device flag.
Supports **float32** and **bfloat16** precision via --precision flag.

Usage:
  python -m vggttt_jax.run_inference_jax \
      --weights vggttt_jax/vggttt.msgpack \
      --images data/nerf_real_360/pinecone/images_8/IMG_7238.png \
               data/nerf_real_360/pinecone/images_8/IMG_7239.png \
      --device cpu \
      --precision float32
"""

import sys
from pathlib import Path

# Add the repository root directory to sys.path to support importing vggttt_jax
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Restrict to single arena for memory reduction on multi-core systems
from vggttt_jax.memory_utils import setup_malloc_arena, trim_memory
setup_malloc_arena()

import argparse
import os
import time
import math
from pathlib import Path

import numpy as np
from PIL import Image


def load_and_preprocess_images_np(
    image_paths: list[str],
    target_size: int = 518,
    patch_size: int = 14,
    mode: str = "crop",
) -> np.ndarray:
    """Load and preprocess images into a NumPy array [S, H, W, 3] in [0, 1].

    Mirrors the PyTorch ``load_and_preprocess_images`` function but uses
    PIL + NumPy only (no torch dependency).

    Args:
        image_paths: List of file paths to images.
        target_size: Target resolution (default 518 for ViT-L/14).
        patch_size: Patch size for divisibility (default 14).
        mode: "crop" or "pad".

    Returns:
        np.ndarray of shape [S, H, W, 3] with float32 values in [0, 1].
    """
    images = []
    for path in image_paths:
        img = Image.open(path)
        if img.mode == "RGBA":
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / patch_size) * patch_size
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / patch_size) * patch_size
        else:  # crop
            new_width = target_size
            new_height = round(height * (new_width / width) / patch_size) * patch_size

        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 3]

        # Center crop height if needed
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            arr = arr[start_y : start_y + target_size, :, :]

        images.append(arr)

    # Check shapes and pad if different
    shapes = {a.shape[:2] for a in images}
    if len(shapes) > 1:
        max_h = max(s[0] for s in shapes)
        max_w = max(s[1] for s in shapes)
        padded = []
        for a in images:
            h, w = a.shape[:2]
            ph = max_h - h
            pw = max_w - w
            if ph > 0 or pw > 0:
                a = np.pad(
                    a,
                    ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0)),
                    mode="constant",
                    constant_values=1.0,
                )
            padded.append(a)
        images = padded

    return np.stack(images, axis=0)  # [S, H, W, 3]


def load_weights_and_cast(path: str, dtype=None):
    """Load weights from msgpack or safetensors, casting to dtype on-the-fly to save RAM."""
    import gc
    import jax.numpy as jnp

    if path.endswith(".safetensors"):
        from safetensors.numpy import load_file
        # Memory-mapped flat dictionary
        flat_variables = load_file(path)
        
        variables = {}
        # Iterate over keys, pop each element to allow immediate GC of the numpy reference
        for k in list(flat_variables.keys()):
            v = flat_variables.pop(k)
            # Convert to JAX array and cast to dtype if specified
            if dtype is not None and hasattr(v, "dtype") and v.dtype != dtype:
                v_jax = jnp.array(v, dtype=dtype)
            else:
                v_jax = jnp.array(v)
            
            # Place in nested structure
            parts = k.split(".")
            current = variables
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = v_jax
            
            # Free variables
            v = None
            v_jax = None
            
        gc.collect()
        return variables
    else:
        # For msgpack, load normally and cast in-place
        from flax import serialization
        with open(path, "rb") as f:
            raw_bytes = f.read()
        variables = serialization.from_bytes(None, raw_bytes)
        
        if dtype is not None:
            def _cast_inplace(d):
                if not isinstance(d, dict):
                    return d
                for k in list(d.keys()):
                    v = d.pop(k)
                    if isinstance(v, dict):
                        d[k] = _cast_inplace(v)
                    elif hasattr(v, "dtype"):
                        d[k] = jnp.array(v, dtype=dtype)
                    else:
                        d[k] = v
                return d
            variables = _cast_inplace(variables)
            gc.collect()
            
        return variables


def main():
    parser = argparse.ArgumentParser(description="VGG-T³ JAX inference runner")
    parser.add_argument(
        "--weights",
        type=str,
        default="vggttt_jax/vggttt.msgpack",
        help="Path to .msgpack weight file",
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        required=True,
        help="Paths to input images (PNG/JPG)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="JAX backend device",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        choices=["float32", "bfloat16"],
        help="Compute precision",
    )
    parser.add_argument(
        "--low-ram",
        action="store_true",
        help="Enables memory-optimized execution (disable JIT, load float16/bfloat16 weights, and perform aggressive memory trims)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="jax_inference_output.npz",
        help="Output .npz path for predictions",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=518,
        help="Image target resolution (default: 518)",
    )
    args = parser.parse_args()

    # If low-ram and weights is default, switch to float16 safetensors weights
    if args.low_ram and args.weights == "vggttt_jax/vggttt.msgpack":
        args.weights = "vggttt_jax/vggttt_f16.safetensors"

    # ---- 0. Configure JAX backend ----
    os.environ["JAX_PLATFORMS"] = args.device
    import jax
    import jax.numpy as jnp

    if args.low_ram:
        jax.config.update('jax_disable_jit', True)
        print("Low-RAM mode: JAX JIT compilation disabled (eager execution).")

    print(f"JAX devices: {jax.devices()}")
    dtype = jnp.float32 if args.precision == "float32" else jnp.bfloat16

    # ---- 1. Load & preprocess images ----
    print(f"Loading {len(args.images)} images...")
    images_np = load_and_preprocess_images_np(args.images, target_size=args.target_size)
    S, H, W, C = images_np.shape
    print(f"  Preprocessed shape: [{S}, {H}, {W}, {C}]")

    # Add batch dimension → [1, S, H, W, 3]
    images_jax = jnp.array(images_np[np.newaxis, ...], dtype=dtype)

    # ---- 2. Load weights ----
    print(f"Loading weights from {args.weights}...")
    t0 = time.time()
    variables = load_weights_and_cast(args.weights, dtype=dtype)
    print(f"  Weights loaded in {time.time() - t0:.1f}s")
    trim_memory()

    # ---- 3. Instantiate model ----
    from vggttt_jax.models import VGGT

    model = VGGT()

    # ---- 4. Run inference ----
    print("Running inference...")
    t_start = time.time()

    if args.low_ram:
        preds = model.apply(variables, images_jax)
    else:
        @jax.jit
        def forward(variables, images):
            return model.apply(variables, images)
        preds = forward(variables, images_jax)

    # Block until computation is done (JAX is async by default)
    jax.block_until_ready(preds)
    t_end = time.time()

    print(f"  Inference completed in {t_end - t_start:.2f}s")

    # ---- 5. Print output shapes ----
    print("\n  Output shapes:")
    for key, val in preds.items():
        print(f"    {key:15s}: {val.shape}")

    # ---- 6. Save outputs ----
    out = {k: np.array(v) for k, v in preds.items()}
    np.savez(args.output, **out)
    print(f"\n  Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
