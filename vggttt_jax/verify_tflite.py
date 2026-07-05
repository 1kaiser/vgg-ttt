"""Verify the converted .tflite model against the original JAX model on real images."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
import time
import numpy as np
import tensorflow as tf

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from vggttt_jax.models import VGGT
from vggttt_jax.run_inference_jax import load_weights_and_cast, load_and_preprocess_images_np

S, H, W = 2, 392, 518
WEIGHTS_PATH = os.path.join(_HERE, "vggttt_f16.safetensors")
TFLITE_PATH = os.path.join(_HERE, "vggttt_dynrange.tflite")
REPO_ROOT = os.path.dirname(_HERE)

image_paths = [
    os.path.join(REPO_ROOT, "data/nerf_real_360/pinecone/images_8/IMG_7238.png"),
    os.path.join(REPO_ROOT, "data/nerf_real_360/pinecone/images_8/IMG_7239.png"),
]
images_np = load_and_preprocess_images_np(image_paths, target_size=518)
assert images_np.shape == (S, H, W, 3), images_np.shape
images = images_np[None].astype(np.float32)  # (1, S, H, W, 3)
print(f"input shape: {images.shape}")

# --- JAX reference ---
print("Running JAX reference...")
model = VGGT()
variables = load_weights_and_cast(WEIGHTS_PATH, dtype=np.float32)
t0 = time.time()
jax_out = model.apply(variables, images)
jax_time = time.time() - t0
print(f"  JAX inference: {jax_time:.1f}s")

# --- TFLite ---
print("Running TFLite...")
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
t0 = time.time()
interpreter.set_tensor(input_details[0]["index"], images)
interpreter.invoke()
tflite_time = time.time() - t0
print(f"  TFLite inference: {tflite_time:.1f}s")

tflite_out = {}
for od in output_details:
    val = interpreter.get_tensor(od["index"])
    for name in ("pose", "intrinsics", "pts3d", "conf", "depth", "depth_conf"):
        expected_shape = np.array(jax_out[name]).shape
        if val.shape == expected_shape and name not in tflite_out:
            tflite_out[name] = val
            break

print(f"\n{'output':<14} {'max abs diff':>14} {'cos sim':>10}")
for name in ("pose", "intrinsics", "pts3d", "conf", "depth", "depth_conf"):
    a = np.array(jax_out[name]).astype(np.float64)
    b = tflite_out[name].astype(np.float64)
    diff = np.abs(a - b).max()
    cos = float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    print(f"{name:<14} {diff:>14.6f} {cos:>10.6f}")

print(f"\nJAX: {jax_time:.1f}s   TFLite: {tflite_time:.1f}s")
print("\nDone.")
