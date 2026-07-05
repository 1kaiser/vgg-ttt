"""Verify the exported ONNX model against the PyTorch eager reference.

Compares onnxruntime (CPUExecutionProvider) output against the same
VGGTONNXWrapper run eagerly on the same weights/images -- isolating
whether the ONNX export itself introduced any numerical drift, separate
from the already-established JAX/TFLite cross-framework comparison.
"""
import argparse
import time

import numpy as np
import onnxruntime as ort
import torch

from convert_to_onnx import VGGTONNXWrapper, OUTPUT_NAMES, _patch_make_sincos_pos_embed
from vggttt.nets.vggt.models.vggt import VGGT
from vggttt.nets.vggt.img import load_and_preprocess_images


def main():
    _patch_make_sincos_pos_embed()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="nvidia/vgg-ttt")
    parser.add_argument("--onnx-path", default="vggttt_jax/vggttt_pt.onnx")
    parser.add_argument(
        "--images", nargs="+",
        default=[
            "data/nerf_real_360/pinecone/images_8/IMG_7238.png",
            "data/nerf_real_360/pinecone/images_8/IMG_7239.png",
        ],
    )
    args = parser.parse_args()

    print(f"Loading PyTorch model from {args.checkpoint} (CPU, float32 eager reference)...")
    model = VGGT.from_pretrained(args.checkpoint).to("cpu").eval()
    wrapper = VGGTONNXWrapper(model, num_ttt_steps=2).to("cpu").eval()

    images = load_and_preprocess_images(args.images)
    print(f"input shape: {tuple(images.shape)}")

    print("Running PyTorch eager (CPU) reference...")
    t0 = time.time()
    with torch.no_grad():
        pt_out = wrapper(images)
    pt_time = time.time() - t0
    print(f"  PyTorch eager (CPU): {pt_time:.1f}s")

    print(f"Loading ONNX model from {args.onnx_path} via onnxruntime (CPUExecutionProvider)...")
    sess = ort.InferenceSession(args.onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    t0 = time.time()
    onnx_out = sess.run(None, {input_name: images.numpy()})
    onnx_time = time.time() - t0
    print(f"  onnxruntime (CPU): {onnx_time:.1f}s")

    # onnxruntime's output order should match output_names passed at export
    # time, but map by name defensively via the session's own metadata.
    onnx_names = [o.name for o in sess.get_outputs()]
    onnx_by_name = dict(zip(onnx_names, onnx_out))

    print(f"\n{'output':<14} {'max abs diff':>14} {'cos sim':>10}")
    for name, pt_t in zip(OUTPUT_NAMES, pt_out):
        a = pt_t.detach().cpu().numpy().astype(np.float64)
        b = onnx_by_name[name].astype(np.float64)
        diff = np.abs(a - b).max()
        cos = float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        print(f"{name:<14} {diff:>14.6f} {cos:>10.6f}")

    print(f"\nPyTorch (CPU): {pt_time:.1f}s   onnxruntime (CPU): {onnx_time:.1f}s")


if __name__ == "__main__":
    main()
