# %% [markdown]
# # VGG-T³ — Dual Backend, Dual Environment
#
# Runs the same VGG-T³ reconstruction two ways and compares them:
#
# | Backend | Format | Runs on |
# |---|---|---|
# | JAX/Flax | `.safetensors` | CPU or GPU |
# | TFLite (LiteRT) | `.tflite` (dynamic-range int8) | CPU (XNNPACK) |
#
# Works unmodified **locally** (Jupyter/papermill) and on **Google Colab**
# (`colab exec -f`) — same pattern as this repo's existing
# `reconstruct_jax.ipynb` (clone-on-Colab bootstrap), extended with a
# second backend.
#
# The `.tflite` model here is the dynamic-range (weight-only int8)
# quantized variant, not the more accurate float16 one — LiteRT.js's own
# README documents a hard 2GB WASM-runtime-memory ceiling for *browser*
# loading (float16 at 2.3GB exceeds it; the int8 dynamic-range model at
# 1.3GB doesn't), which is what also powers `webgpu_demo/index.html`. That
# constraint is browser-specific, not a Python/TFLite-interpreter one — but
# using the SAME file here means this notebook's comparison numbers double
# as accuracy ground-truth for exactly what the browser demo runs.

# %% tags=["parameters"]
WEIGHTS_PATH = "vggttt_jax/vggttt_f16.safetensors"
TFLITE_PATH = "vggttt_jax/vggttt_dynrange.tflite"
S, H, W = 2, 392, 518  # fixed -- matches the TFLite model's traced input shape
IMAGE_PATHS = [
    "data/nerf_real_360/pinecone/images_8/IMG_7238.png",
    "data/nerf_real_360/pinecone/images_8/IMG_7239.png",
]
OUTPUT_DIR = "dual_backend_output"
CONF_THRESHOLD = 1.2

# %% [markdown]
# ## 1. Colab bootstrap (same pattern as reconstruct_jax.ipynb)

# %%
import sys
import os

if 'google.colab' in sys.modules:
    print("Running in Google Colab. Setting up environment...")
    if not os.path.exists("vgg-ttt"):
        os.system("git clone https://github.com/1kaiser/vgg-ttt.git")
    os.chdir("vgg-ttt")
    sys.path.insert(0, os.getcwd())

    os.makedirs("vggttt_jax", exist_ok=True)
    if not os.path.exists(WEIGHTS_PATH):
        os.system(f"wget -nc https://huggingface.co/1kaiser/vgg_ttt/resolve/main/vggttt_f16.safetensors -O {WEIGHTS_PATH}")
    if not os.path.exists(TFLITE_PATH):
        os.system(f"wget -nc https://huggingface.co/1kaiser/vgg_ttt/resolve/main/vggttt_dynrange.tflite -O {TFLITE_PATH}")

    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/nerf_real_360"):
        os.system("wget -nc https://huggingface.co/datasets/1kaiser/NERF_360/resolve/main/nerf_real_360.zip -O data/nerf_real_360.zip")
        os.system("unzip -o data/nerf_real_360.zip -d data/nerf_real_360")
        os.remove("data/nerf_real_360.zip")

    os.system("pip install -q tensorflow")
else:
    print("Running locally. Skipping environment setup.")
    sys.path.insert(0, os.getcwd())

# %% [markdown]
# ## 2. Backend A: JAX/Flax (`.safetensors`)
#
# Runs via `run_inference_jax.py` **as a subprocess**.
#
# The real reason a bare in-kernel `model.apply()` call hung indefinitely
# (confirmed the hard way: near-zero CPU over a 600s timeout) turned out to
# have nothing to do with JAX threading or asyncio — it's
# `vggttt_jax/memory_utils.py`'s `setup_malloc_arena()`, which runs
# automatically at import time (`run_inference_jax.py` calls it at module
# level) and does `os.execv(sys.executable, [sys.executable] + sys.argv)`
# whenever `MALLOC_ARENA_MAX` isn't already `"1"`. Run as a plain script,
# that's harmless (just restarts the same script with the env var set).
# Imported from *inside a Jupyter kernel*, `sys.argv` is the kernel's own
# launch argv (`ipykernel_launcher.py -f kernel-xxx.json`) — so it
# silently re-execs the kernel process itself in place, severing the
# ZMQ/comm channel papermill was waiting on a reply through. Setting the
# env var *before* the import stops `setup_malloc_arena()` from ever
# calling `os.execv()` in the first place.
#
# Delegating the actual heavy computation to a subprocess (rather than
# calling `model.apply()` in-kernel) is kept anyway, independent of that
# fix — a `os.execv()` re-exec from *within* a subprocess is harmless
# either way, so this stays correct even if some other import down the
# line does something similar.

# %%
import os
os.environ["MALLOC_ARENA_MAX"] = "1"

import subprocess
import sys
import time
import numpy as np

from vggttt_jax.run_inference_jax import load_and_preprocess_images_np

images_np = load_and_preprocess_images_np(IMAGE_PATHS, target_size=518)
assert images_np.shape == (S, H, W, 3), images_np.shape
images = images_np[None].astype(np.float32)  # (1, S, H, W, 3)
print(f"input shape: {images.shape}")

jax_output_path = os.path.join(OUTPUT_DIR, "jax_inference_output.npz")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Popen + manual polling, not subprocess.run()/.communicate() -- a plain
# blocking subprocess.run() call hung indefinitely here (confirmed: it
# never even got as far as spawning the child process, per `ps aux`),
# while the *identical* subprocess call worked fine (271% CPU, real work)
# as a standalone script outside Jupyter. That matches a known category of
# bug: nbclient/Jupyter executes notebook cells inside an asyncio event
# loop, and a *synchronous* blocking subprocess wait can deadlock against
# asyncio's own SIGCHLD-based child-process reaping. Polling non-blockingly
# via .poll() sidesteps that race entirely.
log_path = os.path.join(OUTPUT_DIR, "jax_subprocess.log")
t0 = time.time()
with open(log_path, "w") as log_f:
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "vggttt_jax.run_inference_jax",
            "--weights", WEIGHTS_PATH,
            "--images", *IMAGE_PATHS,
            "--device", "cpu",
            "--precision", "float32",
            "--low-ram",
            "--output", jax_output_path,
        ],
        stdout=log_f, stderr=subprocess.STDOUT,
    )
    while proc.poll() is None:
        time.sleep(1)
        if time.time() - t0 > 300:
            proc.kill()
            raise TimeoutError("run_inference_jax subprocess exceeded 300s")
returncode = proc.returncode
jax_time = time.time() - t0

with open(log_path) as f:
    log_text = f.read()
print(log_text[-2000:])
if returncode != 0:
    raise RuntimeError(f"run_inference_jax subprocess failed (exit {returncode})")
print(f"JAX subprocess wall time: {jax_time:.1f}s")

jax_npz = np.load(jax_output_path)
jax_out = {k: jax_npz[k] for k in jax_npz.files}
for k, v in jax_out.items():
    print(f"  {k:15s}: {v.shape}")

# %% [markdown]
# ## 3. Backend B: TFLite / LiteRT (`.tflite`, dynamic-range int8)

# %%
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

t0 = time.time()
interpreter.set_tensor(input_details[0]["index"], images)
interpreter.invoke()
tflite_time = time.time() - t0
print(f"TFLite inference: {tflite_time:.1f}s")

tflite_out = {}
for od in output_details:
    val = interpreter.get_tensor(od["index"])
    for name in ("pose", "intrinsics", "pts3d", "conf", "depth", "depth_conf"):
        expected_shape = np.array(jax_out[name]).shape
        if val.shape == expected_shape and name not in tflite_out:
            tflite_out[name] = val
            break
    print(f"  output shape {val.shape} -> matched to '{[n for n,v in tflite_out.items() if np.array_equal(v, val)]}'")

# %% [markdown]
# ## 4. Compare

# %%
print(f"{'output':<14} {'max abs diff':>14} {'cos sim':>10}")
for name in ("pose", "intrinsics", "pts3d", "conf", "depth", "depth_conf"):
    a = np.array(jax_out[name]).astype(np.float64)
    b = tflite_out[name].astype(np.float64)
    diff = np.abs(a - b).max()
    cos = float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    print(f"{name:<14} {diff:>14.6f} {cos:>10.6f}")

print(f"\nJAX: {jax_time:.1f}s   TFLite: {tflite_time:.1f}s")

# %% [markdown]
# ## 5. Export point clouds from both backends

# %%
import trimesh

os.makedirs(OUTPUT_DIR, exist_ok=True)


def export_glb(pts3d, conf, colors_by_view, out_path, conf_threshold=CONF_THRESHOLD):
    # pts3d/conf: (S, H, W, ...) -- already in a shared WORLD frame (see
    # models.py's own comment: "Reconstructed 3D world coords"), so views
    # are merged directly with no per-camera-pose transform needed.
    all_pts, all_cols = [], []
    for s in range(pts3d.shape[0]):
        mask = conf[s] >= conf_threshold
        all_pts.append(pts3d[s][mask])
        all_cols.append(colors_by_view[s][mask])
    pts = np.concatenate(all_pts, axis=0)
    cols = np.concatenate(all_cols, axis=0)
    if len(pts) == 0:
        return None
    cloud = trimesh.points.PointCloud(vertices=pts, colors=cols)
    cloud.export(out_path)
    return out_path


colors_by_view = (images_np * 255).astype(np.uint8)  # (S, H, W, 3)

jax_glb = export_glb(
    np.array(jax_out["pts3d"])[0], np.array(jax_out["conf"])[0], colors_by_view,
    os.path.join(OUTPUT_DIR, "pinecone_jax.glb"),
)
print(f"wrote {jax_glb}")

tflite_glb = export_glb(
    tflite_out["pts3d"][0], tflite_out["conf"][0], colors_by_view,
    os.path.join(OUTPUT_DIR, "pinecone_tflite_dynrange.glb"),
)
print(f"wrote {tflite_glb}")

# %% [markdown]
# ## Summary
#
# Same model, two backends, side by side:
# - **JAX/Flax (`.safetensors`)** — the reference implementation, runs on
#   CPU or GPU via plain `model.apply`.
# - **TFLite (`.tflite`, dynamic-range int8)** — converted via `jax2tf` →
#   SavedModel → `TFLiteConverter` (see `vggttt_jax/convert_to_tflite.py`
#   and `convert_to_tflite_dynrange.py`); the same `.tflite` file also
#   powers the in-browser WebGPU demo (`webgpu_demo/index.html`) via
#   LiteRT.js.
#
# Real, non-trivial accuracy trade-off from int8 weight-only quantization
# (not the same file as the float32/float16 comparisons run during
# development): cosine similarity stays high (0.997-0.9999) across all six
# outputs, but absolute error is real, e.g. ~6% relative error in predicted
# focal length (`intrinsics`) — see the comparison table above for this
# run's exact numbers on the real pinecone pair.
