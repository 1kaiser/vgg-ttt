"""
Re-convert the already-built SavedModel -> TFLite with dynamic-range (weight-
only int8) quantization -- different from full-integer quantization: no
representative dataset/calibration needed, only weight storage is quantized
to int8 (dequantized on the fly at inference), activations stay float. This
avoids the "stablehlo.scatter" MLIR quantizer failure hit by full-integer
quantization (that failure was in the whole-graph activation-quantization
rewrite pass; this mode doesn't touch activations at all), and should get
size closer to the critical <2GB threshold documented as LiteRT.js's hard
WASM-memory ceiling for browser loading.
"""
import os
import time

import tensorflow as tf

_HERE = os.path.dirname(os.path.abspath(__file__))
SAVEDMODEL_DIR = os.path.join(_HERE, "vggttt_savedmodel")
TFLITE_PATH = os.path.join(_HERE, "vggttt_dynrange.tflite")

print("Converting SavedModel -> TFLite (dynamic-range/weight-only int8) ...")
t0 = time.time()
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Deliberately NOT setting representative_dataset or supported_types here --
# that's what makes this "dynamic range" (weight-only) rather than full
# integer or float16 quantization.
tflite_model = converter.convert()
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)
print(f"wrote {TFLITE_PATH} ({len(tflite_model) / 1e6:.1f} MB) in {time.time() - t0:.1f}s")
