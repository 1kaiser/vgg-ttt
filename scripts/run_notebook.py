import os
import sys
import subprocess
import argparse
import papermill as pm

def main():
    parser = argparse.ArgumentParser(description="Convert and run VGG-T3 reconstruction notebook using Jupytext & Papermill API")
    parser.add_argument("--images", nargs="+", required=True, help="List of image file paths")
    parser.add_argument("--model", type=str, default="nvidia/vgg-ttt", help="Hugging Face model repository")
    parser.add_argument("--conf-threshold", type=float, default=1.2, help="Confidence threshold for filtering point cloud")
    parser.add_argument("--max-points", type=int, default=50000, help="Max points to plot in Plotly Scatter3d")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu)")
    parser.add_argument("--output-notebook", type=str, default="notebooks/reconstruct_output.ipynb", help="Output notebook path")
    
    args = parser.parse_args()

    # Path to jupytext in current conda environment
    python_bin = sys.executable
    conda_env_dir = os.path.dirname(python_bin)
    jupytext_bin = os.path.join(conda_env_dir, "jupytext")

    if not os.path.exists(jupytext_bin):
        jupytext_bin = "jupytext"  # Fallback to PATH

    notebooks_dir = "notebooks"
    os.makedirs(notebooks_dir, exist_ok=True)

    template_py = os.path.join(notebooks_dir, "reconstruct.py")
    template_ipynb = os.path.join(notebooks_dir, "reconstruct.ipynb")

    if not os.path.exists(template_py):
        print(f"Error: Jupytext template file not found at {template_py}")
        sys.exit(1)

    # 1. Convert Jupytext python script to template ipynb
    print(f"Converting Jupytext script '{template_py}' to '{template_ipynb}'...")
    conv_cmd = [jupytext_bin, "--to", "ipynb", template_py, "-o", template_ipynb]
    res = subprocess.run(conv_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Error converting Jupytext template:")
        print(res.stderr)
        sys.exit(1)
    print("Conversion successful.")

    # 2. Run the notebook using papermill Python API
    print(f"Executing notebook using Papermill Python API (writing results to '{args.output_notebook}')...")
    
    try:
        pm.execute_notebook(
            input_path=template_ipynb,
            output_path=args.output_notebook,
            parameters=dict(
                image_paths=args.images,
                model_path=args.model,
                conf_threshold=args.conf_threshold,
                max_points=args.max_points,
                device=args.device
            ),
            kernel_name="num_gpu",
            log_output=True
        )
    except Exception as e:
        print("\nError executing notebook via Papermill API:")
        print(e)
        sys.exit(1)
        
    print("\n" + "="*60)
    print(f"Notebook pipeline execution COMPLETED successfully!")
    print(f"Output saved to: {args.output_notebook}")
    print("="*60)

if __name__ == "__main__":
    main()
