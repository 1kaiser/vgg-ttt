# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # VGG-T³ JAX 3D Reconstruction and Resource Profiling Notebook
# This notebook loads input images, performs 3D reconstruction using the JAX port of VGG-T³, and profiles CPU/RAM usage.

# %% tags=["parameters"]
image_paths = []
weights_path = "vggttt_jax/vggttt_f16.safetensors"
precision = "bfloat16"
low_ram = True
device = "cpu"
conf_threshold = 1.2
max_points = 50000

# %%
import os
import sys
from pathlib import Path

# Add repo root to path dynamically
if "__file__" in locals():
    notebook_dir = Path(__file__).resolve().parent
else:
    cwd = Path(os.getcwd())
    if cwd.name == "notebooks":
        notebook_dir = cwd
    else:
        notebook_dir = cwd / "notebooks"

repo_root = notebook_dir.parent
repo_root_str = str(repo_root)

if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

# Setup malloc arena for low-ram mode
if low_ram:
    os.environ["MALLOC_ARENA_MAX"] = "1"

os.environ["JAX_PLATFORMS"] = device

import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import jax
import jax.numpy as jnp
from vggttt_jax.models import VGGT
from vggttt_jax.run_inference_jax import load_and_preprocess_images_np, load_weights_and_cast
from vggttt_jax.memory_utils import trim_memory

# %%
# Metrics logging setup
process = psutil.Process(os.getpid())
start_time = time.perf_counter()
start_cpu_time = time.process_time()

def log_metrics(label: str):
    elapsed = time.perf_counter() - start_time
    cpu_elapsed = time.process_time() - start_cpu_time
    rss = process.memory_info().rss / (1024 ** 3) # GB
    print(f"[{label}] Elapsed: {elapsed:.2f}s | CPU Time: {cpu_elapsed:.2f}s | Process RAM: {rss:.3f} GB")

log_metrics("Notebook Start")

# %% [markdown]
# ## Load and Preprocess Images
# Input images are preprocessed into NHWC format.

# %%
if not image_paths:
    # Use default pinecone images if none provided
    image_paths = [
        "data/nerf_real_360/pinecone/images_8/IMG_7238.png",
        "data/nerf_real_360/pinecone/images_8/IMG_7239.png"
    ]

# Resolve paths relative to repo_root if they are relative
resolved_paths = []
for p in image_paths:
    path_obj = Path(p)
    if not path_obj.is_absolute():
        resolved_paths.append(str(repo_root / p))
    else:
        resolved_paths.append(p)

print(f"Loading {len(resolved_paths)} images...")
for p in resolved_paths:
    print(f"  - {p}")

images_np = load_and_preprocess_images_np(resolved_paths, target_size=518)
dtype = jnp.float32 if precision == "float32" else jnp.bfloat16
images_jax = jnp.array(images_np[np.newaxis, ...], dtype=dtype)
log_metrics("Images Preprocessed")
print(f"Preprocessed tensor shape: {images_jax.shape}")

# %% [markdown]
# ## Load JAX Weights
# Load the converted Flax weights on-the-fly.

# %%
resolved_weights_path = weights_path
if not Path(resolved_weights_path).is_absolute():
    resolved_weights_path = str(repo_root / weights_path)

print(f"Loading weights from {resolved_weights_path}...")
variables = load_weights_and_cast(resolved_weights_path, dtype=dtype)
log_metrics("Weights Loaded")
trim_memory()

# %% [markdown]
# ## Initialize Model and Configure JIT

# %%
model = VGGT()
if low_ram:
    jax.config.update('jax_disable_jit', True)
    print("Low-RAM mode enabled: JAX JIT disabled (eager execution)")
else:
    print("JAX JIT enabled (first run will compile)")

# %% [markdown]
# ## Run Model Inference

# %%
print("Running JAX VGG-T3 inference...")
t_inference_start = time.perf_counter()

if low_ram:
    preds = model.apply(variables, images_jax)
else:
    @jax.jit
    def forward(vars_dict, imgs):
        return model.apply(vars_dict, imgs)
    preds = forward(variables, images_jax)

jax.block_until_ready(preds)
t_inference_end = time.perf_counter()
log_metrics("Inference Finished")
print(f"Inference Time: {t_inference_end - t_inference_start:.2f} seconds")
trim_memory()

# %% [markdown]
# ## Resource Consumption Summary

# %%
final_rss = process.memory_info().rss / (1024 ** 3)
cpu_percent = psutil.cpu_percent(interval=0.5)

print("="*60)
print("             RESOURCE PROFILING RESULTS (JAX)")
print("="*60)
print(f"Total Execution Time:        {time.perf_counter() - start_time:.2f} seconds")
print(f"Total CPU Process Time:      {time.process_time() - start_cpu_time:.2f} seconds")
print(f"System CPU Usage (current):  {cpu_percent:.1f}%")
print(f"Peak Process RAM (RSS):      {final_rss:.3f} GB")
print("="*60)

# %% [markdown]
# ## Output Analysis and Visualizations

# %%
# Convert outputs back to CPU/NumPy for plotting
img_vis = images_np
# predictions shapes: conf is [1, S, H, W], depth is [1, S, H, W, 1], pts3d is [1, S, H, W, 3]
pts3d = np.array(preds['pts3d'][0])
conf = np.array(preds['conf'][0])
depth = np.array(preds['depth'][0])
poses = np.array(preds['pose'][0])
intrinsics = np.array(preds['intrinsics'][0])

num_images = img_vis.shape[0]

# %%
# Plot Input Images, Depth Maps, and Confidence Maps
fig, axes = plt.subplots(num_images, 3, figsize=(15, 4 * num_images))
if num_images == 1:
    axes = np.expand_dims(axes, axis=0)

for idx in range(num_images):
    # Original Image
    axes[idx, 0].imshow(img_vis[idx])
    axes[idx, 0].set_title(f"Image {idx}")
    axes[idx, 0].axis('off')
    
    # Depth Map
    d_map = depth[idx, ..., 0]
    im_d = axes[idx, 1].imshow(d_map, cmap='turbo')
    axes[idx, 1].set_title(f"Depth {idx}")
    axes[idx, 1].axis('off')
    fig.colorbar(im_d, ax=axes[idx, 1], fraction=0.046, pad=0.04)
    
    # Confidence Map
    c_map = conf[idx]
    im_c = axes[idx, 2].imshow(c_map, cmap='viridis')
    axes[idx, 2].set_title(f"Confidence {idx}")
    axes[idx, 2].axis('off')
    fig.colorbar(im_c, ax=axes[idx, 2], fraction=0.046, pad=0.04)

plt.tight_layout()
assets_dir = repo_root / "assets"
os.makedirs(assets_dir, exist_ok=True)
plt.savefig(assets_dir / "jax_pinecone_reconstruct_output.png", bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Interactive 3D Point Cloud Visualization (Plotly)

# %%
# Flatten and filter by confidence
points_flat = pts3d.reshape(-1, 3)
colors_flat = img_vis.reshape(-1, 3)
conf_flat = conf.reshape(-1)

# Mask out low confidence points
mask = conf_flat > conf_threshold
p_filtered = points_flat[mask]
c_filtered = colors_flat[mask]

print(f"Total points: {len(points_flat)}")
print(f"Points above confidence threshold ({conf_threshold}): {len(p_filtered)}")

# Downsample for rendering performance
if len(p_filtered) > max_points:
    print(f"Downsampling from {len(p_filtered)} to {max_points} for visualization...")
    rng = np.random.default_rng(42)
    indices = rng.choice(len(p_filtered), max_points, replace=False)
    p_filtered = p_filtered[indices]
    c_filtered = c_filtered[indices]

# Map float colors [0, 1] to rgb strings
color_strings = [f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})" for r, g, b in c_filtered]

# Create Plotly interactive 3D Scatter plot
scatter = go.Scatter3d(
    x=p_filtered[:, 0],
    y=p_filtered[:, 1],
    z=p_filtered[:, 2],
    mode='markers',
    marker=dict(
        size=1.2,
        color=color_strings,
        opacity=0.9
    )
)

fig = go.Figure(data=[scatter])

# Configure layout
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode='data'
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    title=f"JAX Port VGG-T³ 3D Point Cloud Reconstruction ({len(p_filtered)} points)"
)

fig.show()

# %% [markdown]
# ## Run Inference on Alternative Dataset (Vasedeck)
# Here we run JAX inference on a different pair of images (from the `vasedeck` dataset) and save the generated reconstruction plots to the `assets` folder.

# %%
vasedeck_image_paths = [
    str(repo_root / "data/nerf_real_360/vasedeck/images/IMG_8361.JPG"),
    str(repo_root / "data/nerf_real_360/vasedeck/images/IMG_8362.JPG")
]

print(f"Loading {len(vasedeck_image_paths)} Vasedeck images...")
vasedeck_images_np = load_and_preprocess_images_np(vasedeck_image_paths, target_size=518)
vasedeck_images_jax = jnp.array(vasedeck_images_np[np.newaxis, ...], dtype=dtype)

print("Running JAX VGG-T3 inference on Vasedeck...")
vasedeck_t_start = time.perf_counter()
if low_ram:
    vasedeck_preds = model.apply(variables, vasedeck_images_jax)
else:
    vasedeck_preds = forward(variables, vasedeck_images_jax)

jax.block_until_ready(vasedeck_preds)
print(f"Vasedeck Inference completed in {time.perf_counter() - vasedeck_t_start:.2f} seconds")
trim_memory()

# %%
# Convert outputs back to CPU/NumPy for plotting
vasedeck_depth = np.array(vasedeck_preds['depth'][0])
vasedeck_conf = np.array(vasedeck_preds['conf'][0])

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for idx in range(2):
    # Original Image
    axes[idx, 0].imshow(vasedeck_images_np[idx])
    axes[idx, 0].set_title(f"Vasedeck Image {idx}")
    axes[idx, 0].axis('off')
    
    # Depth Map
    d_map = vasedeck_depth[idx, ..., 0]
    im_d = axes[idx, 1].imshow(d_map, cmap='turbo')
    axes[idx, 1].set_title(f"Vasedeck Depth {idx}")
    axes[idx, 1].axis('off')
    fig.colorbar(im_d, ax=axes[idx, 1], fraction=0.046, pad=0.04)
    
    # Confidence Map
    c_map = vasedeck_conf[idx]
    im_c = axes[idx, 2].imshow(c_map, cmap='viridis')
    axes[idx, 2].set_title(f"Vasedeck Confidence {idx}")
    axes[idx, 2].axis('off')
    fig.colorbar(im_c, ax=axes[idx, 2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(assets_dir / "jax_vasedeck_reconstruct_output.png", bbox_inches='tight')
plt.show()
