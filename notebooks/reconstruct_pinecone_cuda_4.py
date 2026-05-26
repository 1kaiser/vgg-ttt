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

# %% [markdown] papermill={"duration": 0.001879, "end_time": "2026-05-25T20:08:47.595013+00:00", "exception": false, "start_time": "2026-05-25T20:08:47.593134+00:00", "status": "completed"}
# # VGG-T³ 3D Reconstruction and Resource Profiling Notebook
# This notebook loads input images, performs 3D reconstruction using VGG-T³, and profiles CPU/RAM/VRAM usage.

# %% papermill={"duration": 0.006724, "end_time": "2026-05-25T20:08:47.603307+00:00", "exception": false, "start_time": "2026-05-25T20:08:47.596583+00:00", "status": "completed"} tags=["parameters"]
image_paths = []
model_path = "nvidia/vgg-ttt"
conf_threshold = 1.2
max_points = 50000
device = "cuda"

# %% papermill={"duration": 0.026947, "end_time": "2026-05-25T20:08:47.631620+00:00", "exception": false, "start_time": "2026-05-25T20:08:47.604673+00:00", "status": "completed"} tags=["injected-parameters"]
# Parameters
image_paths = ["/home/kaiser/projects/vgg-ttt/data/nerf_real_360/pinecone/images_8/IMG_7238.png", "/home/kaiser/projects/vgg-ttt/data/nerf_real_360/pinecone/images_8/IMG_7239.png", "/home/kaiser/projects/vgg-ttt/data/nerf_real_360/pinecone/images_8/IMG_7240.png", "/home/kaiser/projects/vgg-ttt/data/nerf_real_360/pinecone/images_8/IMG_7241.png"]
model_path = "nvidia/vgg-ttt"
conf_threshold = 1.2
max_points = 50000
device = "cuda"


# %% papermill={"duration": 2.349767, "end_time": "2026-05-25T20:08:49.984801+00:00", "exception": false, "start_time": "2026-05-25T20:08:47.635034+00:00", "status": "completed"}
import os
import time
import psutil
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from vggttt.nets.vggt.models.vggt import VGGT
from vggttt.nets.vggt.img import load_and_preprocess_images

# %% papermill={"duration": 0.135009, "end_time": "2026-05-25T20:08:50.123645+00:00", "exception": false, "start_time": "2026-05-25T20:08:49.988636+00:00", "status": "completed"}
# Metrics logging setup
process = psutil.Process(os.getpid())
start_time = time.perf_counter()
start_cpu_time = time.process_time()

def log_metrics(label: str):
    elapsed = time.perf_counter() - start_time
    cpu_elapsed = time.process_time() - start_cpu_time
    rss = process.memory_info().rss / (1024 ** 3) # GB
    
    metrics_str = f"[{label}] Elapsed: {elapsed:.2f}s | CPU Time: {cpu_elapsed:.2f}s | Process RAM: {rss:.3f} GB"
    if torch.cuda.is_available():
        vram_alloc = torch.cuda.memory_allocated() / (1024 ** 3) # GB
        vram_max = torch.cuda.max_memory_allocated() / (1024 ** 3) # GB
        metrics_str += f" | VRAM (Alloc/Max): {vram_alloc:.3f} GB / {vram_max:.3f} GB"
    print(metrics_str)

log_metrics("Notebook Start")

# %% [markdown] papermill={"duration": 0.003342, "end_time": "2026-05-25T20:08:50.130883+00:00", "exception": false, "start_time": "2026-05-25T20:08:50.127541+00:00", "status": "completed"}
# ## Load and Preprocess Images
# Input images are preprocessed and formatted as PyTorch tensors.

# %% papermill={"duration": 0.051506, "end_time": "2026-05-25T20:08:50.185886+00:00", "exception": false, "start_time": "2026-05-25T20:08:50.134380+00:00", "status": "completed"}
if not image_paths:
    raise ValueError("Please provide a non-empty list of image_paths in the parameters.")

print(f"Loading {len(image_paths)} images...")
for p in image_paths:
    print(f"  - {p}")

images = load_and_preprocess_images(image_paths)
log_metrics("Images Preprocessed")
print(f"Preprocessed tensor shape: {images.shape}")

# %% [markdown] papermill={"duration": 0.001374, "end_time": "2026-05-25T20:08:50.189125+00:00", "exception": false, "start_time": "2026-05-25T20:08:50.187751+00:00", "status": "completed"}
# ## Initialize VGG-T³ Model
# Loads the pretrained weights from Hugging Face.

# %% papermill={"duration": 9.393053, "end_time": "2026-05-25T20:08:59.583586+00:00", "exception": false, "start_time": "2026-05-25T20:08:50.190533+00:00", "status": "completed"}
print(f"Loading model '{model_path}' on {device}...")
model = VGGT.from_pretrained(model_path)
model = model.to(device).eval()
log_metrics("Model Loaded")

# %% [markdown] papermill={"duration": 0.003592, "end_time": "2026-05-25T20:08:59.591394+00:00", "exception": false, "start_time": "2026-05-25T20:08:59.587802+00:00", "status": "completed"}
# ## Run Model Inference
# Perform 3D reconstruction and measure resource requirements.

# %% papermill={"duration": 0.919212, "end_time": "2026-05-25T20:09:00.514279+00:00", "exception": false, "start_time": "2026-05-25T20:08:59.595067+00:00", "status": "completed"}
# Reset peak VRAM tracking
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

print("Running VGG-T3 inference...")
t_inference_start = time.perf_counter()

images_device = images.to(device)
with torch.no_grad():
    preds = model.infer(images_device)

t_inference_end = time.perf_counter()
log_metrics("Inference Finished")
print(f"Inference Time: {t_inference_end - t_inference_start:.2f} seconds")

# %% [markdown] papermill={"duration": 0.003798, "end_time": "2026-05-25T20:09:00.522456+00:00", "exception": false, "start_time": "2026-05-25T20:09:00.518658+00:00", "status": "completed"}
# ## Resource Consumption Summary

# %% papermill={"duration": 0.510812, "end_time": "2026-05-25T20:09:01.037131+00:00", "exception": false, "start_time": "2026-05-25T20:09:00.526319+00:00", "status": "completed"}
final_rss = process.memory_info().rss / (1024 ** 3)
cpu_percent = psutil.cpu_percent(interval=0.5)

print("="*60)
print("             RESOURCE PROFILING RESULTS")
print("="*60)
print(f"Total Execution Time:        {time.perf_counter() - start_time:.2f} seconds")
print(f"Total CPU Process Time:      {time.process_time() - start_cpu_time:.2f} seconds")
print(f"System CPU Usage (current):  {cpu_percent:.1f}%")
print(f"Peak Process RAM (RSS):      {final_rss:.3f} GB")
if torch.cuda.is_available():
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"Peak GPU VRAM Allocated:     {peak_vram:.3f} GB")
print("="*60)

# %% [markdown] papermill={"duration": 0.003898, "end_time": "2026-05-25T20:09:01.045531+00:00", "exception": false, "start_time": "2026-05-25T20:09:01.041633+00:00", "status": "completed"}
# ## Output Analysis and Visualizations
# Displaying input images, predicted depth maps, and confidence maps.

# %% papermill={"duration": 0.009025, "end_time": "2026-05-25T20:09:01.058500+00:00", "exception": false, "start_time": "2026-05-25T20:09:01.049475+00:00", "status": "completed"}
# Fetch outputs back to CPU/NumPy for visualization
images_np = images.permute(0, 2, 3, 1).cpu().numpy()
pts3d = preds['pts3d'].cpu().numpy()
conf = preds['conf'].cpu().numpy()
depth = preds['depth'].cpu().numpy()
poses = preds['pose'].cpu().numpy()
intrinsics = preds['intrinsics'].cpu().numpy()

num_images = len(image_paths)

# %% papermill={"duration": 0.990597, "end_time": "2026-05-25T20:09:02.053171+00:00", "exception": false, "start_time": "2026-05-25T20:09:01.062574+00:00", "status": "completed"}
# Plot Input Images, Depth Maps, and Confidence Maps
fig, axes = plt.subplots(num_images, 3, figsize=(15, 4 * num_images))
if num_images == 1:
    axes = np.expand_dims(axes, axis=0)

for idx in range(num_images):
    # Original Image
    axes[idx, 0].imshow(images_np[idx])
    axes[idx, 0].set_title(f"Image {idx}")
    axes[idx, 0].axis('off')
    
    # Depth Map
    d_map = depth[idx, ..., 0]
    im_d = axes[idx, 1].imshow(d_map, cmap='spectral' if hasattr(plt.cm, 'spectral') else 'turbo')
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
plt.show()

# %% [markdown] papermill={"duration": 0.015617, "end_time": "2026-05-25T20:09:02.095886+00:00", "exception": false, "start_time": "2026-05-25T20:09:02.080269+00:00", "status": "completed"}
# ## Interactive 3D Point Cloud Visualization (Plotly)
# Renders the reconstructed 3D scene directly in the notebook output.

# %% papermill={"duration": 1.458643, "end_time": "2026-05-25T20:09:03.570928+00:00", "exception": false, "start_time": "2026-05-25T20:09:02.112285+00:00", "status": "completed"}
# Flatten and filter by confidence
points_flat = pts3d.reshape(-1, 3)
colors_flat = images_np.reshape(-1, 3)
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
    title=f"VGG-T³ 3D Point Cloud Reconstruction ({len(p_filtered)} points)"
)

fig.show()
