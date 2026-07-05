"""
Convert VGG-T3 JAX/Flax model -> TFLite, for LiteRT.js (WebGPU) inference.

Pipeline: JAX/Flax -> jax2tf -> TF SavedModel -> TFLiteConverter -> .tflite

Fixed at a single (S, H, W) = (2, 392, 518) input shape -- the exact shape
produced by preprocessing the repo's own pinecone/images_8 pair at the
default target_size=518 -- since a browser demo needs a fixed shape
anyway, and the two flagged-risky resize ops (bicubic-antialiased
positional-embedding interpolation, DPT head scale_and_translate) were
already isolated-tested and convert cleanly (see
test_resize_jax2tf.py in the astro/webgpu session scratchpad).
"""
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from jax.experimental import jax2tf

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from vggttt_jax.models import VGGT
from vggttt_jax.run_inference_jax import load_weights_and_cast

S, H, W = 2, 392, 518
WEIGHTS_PATH = os.path.join(_HERE, "vggttt_f16.safetensors")
SAVEDMODEL_DIR = os.path.join(_HERE, "vggttt_savedmodel")
TFLITE_PATH = os.path.join(_HERE, "vggttt.tflite")

print(f"[1/5] Loading model + weights ({WEIGHTS_PATH}) ...")
t0 = time.time()
model = VGGT()
variables = load_weights_and_cast(WEIGHTS_PATH, dtype=jnp.float32)
dummy = jnp.zeros((1, S, H, W, 3), dtype=jnp.float32)
print(f"      loaded in {time.time() - t0:.1f}s")

print(f"[2/5] Wrapping model.apply as a jax2tf-convertible function at S={S} {H}x{W} ...")

# IMPORTANT: `variables` (~4.84 GB total) must be an *explicit argument* to
# the traced function, not a Python closure -- closing over it makes jax2tf
# bake the entire weight tree into ONE embedded XLA constant, which blows
# past TensorFlow's hard 2GB-per-tensor-proto limit
# ("Cannot create a tensor proto whose content is larger than 2GB").
# MoGe-2 (125MB total) never hit this ceiling; VGG-T3 (~4.84GB) does.
# Fix (the documented jax2tf pattern for model parameters): pass variables
# in, then store each leaf as its own tf.Variable in the tf.Module, so
# SavedModel checkpoints them as separate entries instead of one blob.


def predict(variables, images):
    out = model.apply(variables, images)
    return out["pose"], out["intrinsics"], out["pts3d"], out["conf"], out["depth"], out["depth_conf"]


t0 = time.time()
tf_predict = jax2tf.convert(predict, enable_xla=True, with_gradient=False)
print(f"      jax2tf.convert() call returned in {time.time() - t0:.1f}s (lazy -- real tracing happens on first call)")

print("[3/5] Building tf.Module + saving SavedModel ...")

OUTPUT_NAMES = ["pose", "intrinsics", "pts3d", "conf", "depth", "depth_conf"]

tf_variables = jax.tree_util.tree_map(lambda x: tf.Variable(x, trainable=False), variables)


class VGGTModule(tf.Module):
    def __init__(self, tf_variables):
        super().__init__()
        self._tf_variables = tf_variables
        # tf.Module auto-tracks tf.Variable leaves found in its own __dict__
        # via nested dicts/lists, so the tree above gets checkpointed.

    @tf.function(input_signature=[tf.TensorSpec([1, S, H, W, 3], tf.float32, name="images")])
    def __call__(self, images):
        outs = tf_predict(self._tf_variables, images)
        return dict(zip(OUTPUT_NAMES, outs))


module = VGGTModule(tf_variables)
t0 = time.time()
_ = module(dummy)  # force one trace
print(f"      first trace + eager call completed in {time.time() - t0:.1f}s")

t0 = time.time()
tf.saved_model.save(module, SAVEDMODEL_DIR, signatures={"serving_default": module.__call__})
print(f"      saved to {SAVEDMODEL_DIR}/ in {time.time() - t0:.1f}s")

print("[4/5] Converting SavedModel -> TFLite ...")
t0 = time.time()
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
tflite_model = converter.convert()
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)
print(f"      wrote {TFLITE_PATH} ({len(tflite_model) / 1e6:.1f} MB) in {time.time() - t0:.1f}s")

print("[5/5] Sanity-check: running the .tflite model via tf.lite.Interpreter ...")
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
test_images = np.random.rand(1, S, H, W, 3).astype(np.float32)
t0 = time.time()
interpreter.set_tensor(input_details[0]["index"], test_images)
interpreter.invoke()
print(f"      invoke() completed in {time.time() - t0:.1f}s")
for od in output_details:
    val = interpreter.get_tensor(od["index"])
    print(f"      output '{od['name']}': shape={val.shape} dtype={val.dtype}")

print("\nDone.")
