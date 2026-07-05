"""
Re-convert the already-built SavedModel -> TFLite with full integer (int8)
post-training quantization, calibrated against a representative dataset of
real preprocessed images. Reuses vggttt_savedmodel/ -- skips the slow
JAX->SavedModel tracing step.

Note: each representative-dataset sample requires one real forward pass
(~70-90s on this machine) to observe activation ranges, so the sample count
here is deliberately small (2, the real pinecone pair) rather than the
100+ TFLite normally recommends -- a real limitation of calibrating this
particular (large, slow-per-inference) model cheaply, noted explicitly
rather than hidden.
"""
import os
import time

import numpy as np
import tensorflow as tf

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)
SAVEDMODEL_DIR = os.path.join(_HERE, "vggttt_savedmodel")
TFLITE_PATH = os.path.join(_HERE, "vggttt_int8.tflite")

import sys
sys.path.insert(0, REPO_ROOT)
from vggttt_jax.run_inference_jax import load_and_preprocess_images_np

image_paths = [
    os.path.join(REPO_ROOT, "data/nerf_real_360/pinecone/images_8/IMG_7238.png"),
    os.path.join(REPO_ROOT, "data/nerf_real_360/pinecone/images_8/IMG_7239.png"),
]
images_np = load_and_preprocess_images_np(image_paths, target_size=518)
sample = images_np[None].astype(np.float32)  # (1, 2, 392, 518, 3)
print(f"representative sample shape: {sample.shape}")


def representative_dataset():
    # Real data (the actual pinecone pair) beats random noise for range
    # calibration, but only 1 real sample -- see module docstring.
    yield [sample]


print("Converting SavedModel -> TFLite (int8 quantized, representative-dataset calibrated) ...")
t0 = time.time()
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
# Full-integer-with-float-fallback: ops that support int8 run in int8;
# anything that doesn't (e.g. some SELECT_TF_OPS) falls back to float,
# rather than hard-failing the whole conversion. Keep float32 I/O so the
# demo's JS caller doesn't need to handle a quantized input/output dtype.
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
tflite_model = converter.convert()
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)
print(f"wrote {TFLITE_PATH} ({len(tflite_model) / 1e6:.1f} MB) in {time.time() - t0:.1f}s")
