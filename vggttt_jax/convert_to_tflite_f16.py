"""
Re-convert the already-built SavedModel -> TFLite with float16 post-training
quantization, to shrink the 4.6GB float32 .tflite down (weights were already
f16-sourced in vggttt_f16.safetensors, so this shouldn't cost meaningful
accuracy -- re-verify with verify_tflite.py after this).

Reuses vggttt_savedmodel/ built by convert_to_tflite.py -- skips the slow
JAX->SavedModel tracing step entirely.
"""
import os
import time

import tensorflow as tf

_HERE = os.path.dirname(os.path.abspath(__file__))
SAVEDMODEL_DIR = os.path.join(_HERE, "vggttt_savedmodel")
TFLITE_PATH = os.path.join(_HERE, "vggttt_f16.tflite")

print("Converting SavedModel -> TFLite (float16 quantized) ...")
t0 = time.time()
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)
print(f"wrote {TFLITE_PATH} ({len(tflite_model) / 1e6:.1f} MB) in {time.time() - t0:.1f}s")
