import os
import sys
import numpy as np
import msgpack
from safetensors.numpy import load_file
from huggingface_hub import hf_hub_download
from flax import serialization

def transpose_conv2d(w):
    # PyTorch: [out_channels, in_channels/groups, kH, kW]
    # JAX: [kH, kW, in_channels/groups, out_channels]
    return w.transpose(2, 3, 1, 0)

def convert_weights():
    print("Downloading PyTorch model.safetensors from huggingface...")
    path = hf_hub_download(repo_id="nvidia/vgg-ttt", filename="model.safetensors")
    print(f"Loaded weights path: {path}")

    # Load using safetensors.numpy to avoid PyTorch loading in RAM
    sd = load_file(path)
    print(f"Loaded state dict containing {len(sd)} keys.")

    jax_params = {}

    for key, val in sd.items():
        # Map key name to nested dictionary structure
        parts = key.split(".")
        
        # Determine parameter type and transpose if needed
        name = parts[-1]
        parent_parts = parts[:-1]
        
        # Determine replacement for weight/bias to JAX names
        if name == "weight":
            # Check if this is a ConvTranspose2d layer (resize_layers.0 or resize_layers.1)
            is_conv_transpose = "resize_layers" in parent_parts and any(p in ["0", "1"] for p in parent_parts)
            
            if is_conv_transpose:
                # ConvTranspose2d weight -> transpose to (kH, kW, in, out) and flip spatial dimensions
                val = val.transpose(2, 3, 0, 1)
                val = np.flip(val, axis=(0, 1))
                jax_name = "kernel"
            elif len(val.shape) == 2:
                # Linear layer weight -> transpose
                val = val.T
                jax_name = "kernel"
            elif len(val.shape) == 4:
                # Conv2d layer weight -> transpose
                val = transpose_conv2d(val)
                jax_name = "kernel"
            elif len(val.shape) == 1:
                # Normalization layer scale (LayerNorm) -> keep as scale
                jax_name = "scale"
            else:
                jax_name = "kernel"
        elif name == "bias":
            jax_name = "bias"
        elif name == "gamma":
            # LayerScale gamma -> scale
            jax_name = "scale"
        elif name in ["camera_token", "register_token", "cls_token", "pos_embed", "register_tokens", "empty_pose_tokens", "w0", "w1", "w2"]:
            # Token embeddings / constants / buffers
            jax_name = name
        else:
            jax_name = name

        # Map parent keys to nested dict structure
        # In JAX/Flax, nested modules are nested dictionaries
        # Convert digit parts to index strings (e.g. '0' -> '0')
        current = jax_params
        for p in parent_parts:
            # PyTorch: modules under Sequential are indexed by digit
            if p.isdigit():
                p = f"layers_{p}" # Standardize naming for Sequential layers in Flax
            if p not in current:
                current[p] = {}
            current = current[p]
            
        current[jax_name] = val

    # Wrap in standard 'params' dict for Flax Linen
    params_wrapped = {"params": jax_params}

    if args.precision == "float16":
        # Recursively cast to float16
        def cast_dict(d):
            new_d = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    new_d[k] = cast_dict(v)
                elif isinstance(v, np.ndarray) and v.dtype == np.float32:
                    new_d[k] = v.astype(np.float16)
                else:
                    new_d[k] = v
            return new_d
        params_wrapped = cast_dict(params_wrapped)

    if args.format == "safetensors":
        # Flatten dictionary for safetensors
        def flatten_dict(d, parent_key="", sep="."):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        flat_params = flatten_dict(params_wrapped)
        suffix = "_f16" if args.precision == "float16" else ""
        output_path = f"vggttt_jax/vggttt{suffix}.safetensors"
        print(f"Saving variables to safetensors format at {output_path}...")
        from safetensors.numpy import save_file
        save_file(flat_params, output_path)
    else:
        suffix = "_f16" if args.precision == "float16" else ""
        output_path = f"vggttt_jax/vggttt{suffix}.msgpack"
        print(f"Serializing variables to msgpack format at {output_path}...")
        serialized_bytes = serialization.to_bytes(params_wrapped)
        with open(output_path, "wb") as f:
            f.write(serialized_bytes)
        
    print(f"Weights successfully converted and saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Target precision for converted weights"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="msgpack",
        choices=["msgpack", "safetensors"],
        help="Output weight format"
    )
    args = parser.parse_args()
    convert_weights()


