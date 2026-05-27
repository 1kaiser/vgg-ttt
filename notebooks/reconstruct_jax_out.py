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

# %% [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/1kaiser/vgg-ttt/blob/main/notebooks/reconstruct_jax_out.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] papermill={"duration": 0.001822, "end_time": "2026-05-26T20:52:01.173171+00:00", "exception": false, "start_time": "2026-05-26T20:52:01.171349+00:00", "status": "completed"} id="d7a0c183"
# # VGG-T³ JAX 3D Reconstruction and Resource Profiling Notebook
# This notebook loads input images, performs 3D reconstruction using the JAX port of VGG-T³, and profiles CPU/RAM usage.

# %% colab={"base_uri": "https://localhost:8080/"} id="colab-setup"
import sys
import os

if 'google.colab' in sys.modules:
    print("Running in Google Colab. Setting up environment...")
    if not os.path.exists("vgg-ttt"):
        os.system("git clone https://github.com/1kaiser/vgg-ttt.git")
    os.chdir("vgg-ttt")
    sys.path.insert(0, os.getcwd())
    
    os.makedirs("vggttt_jax", exist_ok=True)
    if not os.path.exists("vggttt_jax/vggttt_f16.safetensors"):
        os.system("wget -nc https://huggingface.co/datasets/1kaiser/vgg_ttt/resolve/main/vggttt_f16.safetensors -O vggttt_jax/vggttt_f16.safetensors")
        
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/nerf_real_360"):
        os.system("wget -nc https://huggingface.co/datasets/1kaiser/NERF_360/resolve/main/nerf_real_360.zip -O data/nerf_real_360.zip")
        os.system("unzip -o data/nerf_real_360.zip -d data/nerf_real_360")
        os.remove("data/nerf_real_360.zip")
else:
    print("Running locally. Skipping environment setup.")

# %% papermill={"duration": 0.006314, "end_time": "2026-05-26T20:52:01.181001+00:00", "exception": false, "start_time": "2026-05-26T20:52:01.174687+00:00", "status": "completed"} tags=["parameters"] id="e1bfc282"
image_paths = []
weights_path = "vggttt_jax/vggttt_f16.safetensors"
precision = "bfloat16"
low_ram = True
device = "cpu"
conf_threshold = 1.2
max_points = 50000

# %% papermill={"duration": 0.660751, "end_time": "2026-05-26T20:52:01.843059+00:00", "exception": false, "start_time": "2026-05-26T20:52:01.182308+00:00", "status": "completed"} id="560b1dee"
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

# %% papermill={"duration": 0.009785, "end_time": "2026-05-26T20:52:01.856482+00:00", "exception": false, "start_time": "2026-05-26T20:52:01.846697+00:00", "status": "completed"} colab={"base_uri": "https://localhost:8080/"} id="538cc0d2" outputId="313f7fdc-e4bd-45ad-f926-60a9985de4d5"
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

# %% [markdown] papermill={"duration": 0.003152, "end_time": "2026-05-26T20:52:01.862978+00:00", "exception": false, "start_time": "2026-05-26T20:52:01.859826+00:00", "status": "completed"} id="0c6e370b"
# ## Load and Preprocess Images
# Input images are preprocessed into NHWC format.

# %% papermill={"duration": 0.174608, "end_time": "2026-05-26T20:52:02.040792+00:00", "exception": false, "start_time": "2026-05-26T20:52:01.866184+00:00", "status": "completed"} colab={"base_uri": "https://localhost:8080/"} id="15752ab2" outputId="828219fc-505e-44ae-f418-92788fa0a36d"
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

# %% [markdown] papermill={"duration": 0.003288, "end_time": "2026-05-26T20:52:02.047905+00:00", "exception": false, "start_time": "2026-05-26T20:52:02.044617+00:00", "status": "completed"} id="e68e20aa"
# ## Load JAX Weights
# Load the converted Flax weights on-the-fly.

# %% papermill={"duration": 5.121812, "end_time": "2026-05-26T20:52:07.173024+00:00", "exception": false, "start_time": "2026-05-26T20:52:02.051212+00:00", "status": "completed"} colab={"base_uri": "https://localhost:8080/"} id="46e4438f" outputId="3fe36434-642a-4886-d08f-7dcce7fd55ac"
resolved_weights_path = weights_path
if not Path(resolved_weights_path).is_absolute():
    resolved_weights_path = str(repo_root / weights_path)

print(f"Loading weights from {resolved_weights_path}...")
variables = load_weights_and_cast(resolved_weights_path, dtype=dtype)
log_metrics("Weights Loaded")
trim_memory()

# %% [markdown] papermill={"duration": 0.003412, "end_time": "2026-05-26T20:52:07.180692+00:00", "exception": false, "start_time": "2026-05-26T20:52:07.177280+00:00", "status": "completed"} id="554b691e"
# ## Initialize Model and Configure JIT

# %% papermill={"duration": 0.009297, "end_time": "2026-05-26T20:52:07.193471+00:00", "exception": false, "start_time": "2026-05-26T20:52:07.184174+00:00", "status": "completed"} colab={"base_uri": "https://localhost:8080/"} id="92fdcdae" outputId="8fd86714-fc9a-48df-9f65-90676c50fb6f"
model = VGGT()
if low_ram:
    jax.config.update('jax_disable_jit', True)
    print("Low-RAM mode enabled: JAX JIT disabled (eager execution)")
else:
    print("JAX JIT enabled (first run will compile)")

# %% [markdown] papermill={"duration": 0.003419, "end_time": "2026-05-26T20:52:07.200744+00:00", "exception": false, "start_time": "2026-05-26T20:52:07.197325+00:00", "status": "completed"} id="fa344965"
# ## Run Model Inference

# %% papermill={"duration": 90.47252, "end_time": "2026-05-26T20:53:37.676802+00:00", "exception": false, "start_time": "2026-05-26T20:52:07.204282+00:00", "status": "completed"} colab={"base_uri": "https://localhost:8080/"} id="72b551d0" outputId="fc3e4702-b280-4773-ab5e-23051bea0073"
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

# %% [markdown] papermill={"duration": 0.003593, "end_time": "2026-05-26T20:53:37.684307+00:00", "exception": false, "start_time": "2026-05-26T20:53:37.680714+00:00", "status": "completed"} id="7b44c67d"
# ## Resource Consumption Summary

# %% papermill={"duration": 0.511436, "end_time": "2026-05-26T20:53:38.199377+00:00", "exception": false, "start_time": "2026-05-26T20:53:37.687941+00:00", "status": "completed"} id="1514b082" colab={"base_uri": "https://localhost:8080/"} outputId="51a728fb-49df-4a75-92d5-b8de429425bf"
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

# %% [markdown] papermill={"duration": 0.003709, "end_time": "2026-05-26T20:53:38.207374+00:00", "exception": false, "start_time": "2026-05-26T20:53:38.203665+00:00", "status": "completed"} id="e8580429"
# ## Output Analysis and Visualizations

# %% papermill={"duration": 0.175513, "end_time": "2026-05-26T20:53:38.386635+00:00", "exception": false, "start_time": "2026-05-26T20:53:38.211122+00:00", "status": "completed"} id="a565d530"
# Convert outputs back to CPU/NumPy for plotting
img_vis = images_np
# predictions shapes: conf is [1, S, H, W], depth is [1, S, H, W, 1], pts3d is [1, S, H, W, 3]
pts3d = np.array(preds['pts3d'][0])
conf = np.array(preds['conf'][0])
depth = np.array(preds['depth'][0])
poses = np.array(preds['pose'][0])
intrinsics = np.array(preds['intrinsics'][0])

num_images = img_vis.shape[0]

# %% papermill={"duration": 0.813625, "end_time": "2026-05-26T20:53:39.204599+00:00", "exception": false, "start_time": "2026-05-26T20:53:38.390974+00:00", "status": "completed"} id="0551bb60" colab={"base_uri": "https://localhost:8080/", "height": 568} outputId="e202c84d-8f0c-43b7-d41e-0915458abb6b"
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

# %% [markdown] papermill={"duration": 0.013907, "end_time": "2026-05-26T20:53:39.236026+00:00", "exception": false, "start_time": "2026-05-26T20:53:39.222119+00:00", "status": "completed"} id="e52ec297"
# ## Interactive 3D Point Cloud Visualization (Plotly)

# %% papermill={"duration": 1.230537, "end_time": "2026-05-26T20:53:40.475283+00:00", "exception": false, "start_time": "2026-05-26T20:53:39.244746+00:00", "status": "completed"} id="98280a77" colab={"base_uri": "https://localhost:8080/", "height": 596} outputId="a5a647df-cb40-4fb6-ea7c-e196754e9687"
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
# ## Export Reconstructed Pinecone Point Cloud to PLY and GLB (Trimesh)
# We use Trimesh to export the filtered 3D point cloud to both PLY and GLB formats.

# %%
import trimesh

# Convert RGB float colors [0, 1] to uint8 [0, 255]
colors_uint8 = (c_filtered * 255).astype(np.uint8)
pc = trimesh.PointCloud(vertices=p_filtered, colors=colors_uint8)

# Save to assets directory
ply_path = assets_dir / "jax_pinecone_reconstruct.ply"
glb_path = assets_dir / "jax_pinecone_reconstruct.glb"
pc.export(ply_path)
pc.export(glb_path)
print(f"Exported Pinecone point cloud to:")
print(f"  - {ply_path}")
print(f"  - {glb_path}")

# %% [markdown] papermill={"duration": 0.021994, "end_time": "2026-05-26T20:53:40.544989+00:00", "exception": false, "start_time": "2026-05-26T20:53:40.522995+00:00", "status": "completed"} id="552d8ef6"
# ## Run Inference on Alternative Dataset (Vasedeck)
# Here we run JAX inference on a different pair of images (from the `vasedeck` dataset) and save the generated reconstruction plots to the `assets` folder.

# %% papermill={"duration": 41.976479, "end_time": "2026-05-26T20:54:22.543300+00:00", "exception": false, "start_time": "2026-05-26T20:53:40.566821+00:00", "status": "completed"} id="85c2cb94" colab={"base_uri": "https://localhost:8080/"} outputId="1a0f44b1-46d0-47e3-b557-02358e987777"
vasedeck_image_paths = [
    str(repo_root / "data/nerf_real_360/vasedeck/images/IMG_8361.JPG"),
    str(repo_root / "data/nerf_real_360/vasedeck/images/IMG_8362.JPG"),
    str(repo_root / "data/nerf_real_360/vasedeck/images/IMG_8363.JPG"),
    str(repo_root / "data/nerf_real_360/vasedeck/images/IMG_8364.JPG"),
    str(repo_root / "data/nerf_real_360/vasedeck/images/IMG_8365.JPG")
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

# %% id="RlKojxXdSi0W"
# Convert outputs back to CPU/NumPy for plotting
img_vis = vasedeck_images_np
# predictions shapes: conf is [1, S, H, W], depth is [1, S, H, W, 1], pts3d is [1, S, H, W, 3]
pts3d = np.array(vasedeck_preds['pts3d'][0])
conf = np.array(vasedeck_preds['conf'][0])
depth = np.array(vasedeck_preds['depth'][0])
poses = np.array(vasedeck_preds['pose'][0])
intrinsics = np.array(vasedeck_preds['intrinsics'][0])

num_images = img_vis.shape[0]

# %% id="qFGQTMwRSl6a" outputId="ebafc1e9-7633-4462-f1d5-729051f2db25" colab={"base_uri": "https://localhost:8080/", "height": 596}
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
# ## Export Reconstructed Vasedeck Point Cloud to PLY and GLB (Trimesh)
# We use Trimesh to export the filtered Vasedeck 3D point cloud to both PLY and GLB formats.

# %%
# Convert RGB float colors [0, 1] to uint8 [0, 255]
vasedeck_colors_uint8 = (c_filtered * 255).astype(np.uint8)
vasedeck_pc = trimesh.PointCloud(vertices=p_filtered, colors=vasedeck_colors_uint8)

# Save to assets directory
vasedeck_ply_path = assets_dir / "jax_vasedeck_reconstruct.ply"
vasedeck_glb_path = assets_dir / "jax_vasedeck_reconstruct.glb"
vasedeck_pc.export(vasedeck_ply_path)
vasedeck_pc.export(vasedeck_glb_path)
print(f"Exported Vasedeck point cloud to:")
print(f"  - {vasedeck_ply_path}")
print(f"  - {vasedeck_glb_path}")

# %% papermill={"duration": 0.797146, "end_time": "2026-05-26T20:54:23.377048+00:00", "exception": false, "start_time": "2026-05-26T20:54:22.579902+00:00", "status": "completed"} id="f32c0287" colab={"base_uri": "https://localhost:8080/", "height": 567} outputId="76b4c30a-160d-429d-bf1d-f4971274909c"
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
