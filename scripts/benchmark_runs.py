import os
import subprocess
import sys

# Define python binary
python_bin = sys.executable

# Images lists
images_base = "/home/kaiser/projects/vgg-ttt/data/nerf_real_360/pinecone/images_8"
images_list = [
    os.path.join(images_base, f"IMG_{i}.png") for i in range(7238, 7244)
]

runs = [
    # CUDA runs only (CPU runs are not supported by the VGG-T3 model codebase due to hardcoded CUDA operations)
    {"name": "cuda_2", "images": images_list[:2], "device": "cuda", "output": "notebooks/reconstruct_pinecone_cuda_2.ipynb"},
    {"name": "cuda_4", "images": images_list[:4], "device": "cuda", "output": "notebooks/reconstruct_pinecone_cuda_4.ipynb"},
    {"name": "cuda_6", "images": images_list[:6], "device": "cuda", "output": "notebooks/reconstruct_pinecone_cuda_6.ipynb"},
]

print("Starting benchmark run suite (CUDA only)...")
for run in runs:
    print(f"\n==========================================")
    print(f"RUNNING BENCHMARK: {run['name'].upper()} on {run['device'].upper()}")
    print(f"==========================================")
    
    cmd = [
        python_bin, "scripts/run_notebook.py",
        "--images"
    ] + run["images"] + [
        "--output-notebook", run["output"],
        "--device", run["device"]
    ]
    
    subprocess.run(cmd)
    
print("\nAll CUDA benchmark runs completed.")
