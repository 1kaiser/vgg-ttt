"""Convert the original PyTorch VGG-T3 checkpoint (nvidia/vgg-ttt) to ONNX.

Bypasses both `VGGT.forward()` (returns a nested/heterogeneous dict, and
skips the attn_kwargs/TTT-step config that `infer()` uses) and
`VGGT.infer()` as-is (its `use_global_pred=False` default path calls
`unproject_depth_map_to_point_map`, which does an explicit `.cpu().numpy()`
detour -- not traceable). Instead, this wrapper calls the same aggregator
-> camera_head -> depth_head -> point_head sequence `infer()` uses, but
sticks to the `point_head`/"global prediction" branch throughout, so the
whole forward pass stays in torch tensor ops end to end.

num_ttt_steps is fixed at 2 (not infer()'s own default of 1) to match the
already-verified JAX port and this checkpoint's own config.json
(global_attn_class.num_steps: 2).
"""
import argparse
import time

import torch
import torch.nn as nn

from vggttt.nets.ttt import TTTOperator
from vggttt.nets.vggt.models.vggt import VGGT
from vggttt.nets.vggt.img import load_and_preprocess_images
from vggttt.nets.vggt.utils.geometry import closed_form_inverse_se3
from vggttt.nets.vggt.utils.pose_enc import pose_encoding_to_extri_intri

OUTPUT_NAMES = ["pose", "intrinsics", "pts3d", "conf", "depth", "depth_conf"]


def _patch_make_sincos_pos_embed():
    """Work around two dynamo-ONNX-exporter / onnxruntime gaps, not numerics bugs.

    `heads/utils.py::make_sincos_pos_embed` builds `omega` as `torch.double`
    while `pos` stays float32, relying on PyTorch's implicit type promotion
    inside `torch.einsum` (silently upcasts `pos` to float64 for the op,
    eager-mode only). Exporting that as-is hits two separate problems:
      1. The dynamo exporter doesn't insert the equivalent `Cast`, so the
         graph ends up with an `Einsum` node whose two inputs have
         different dtypes -- invalid ONNX ("Type parameter (T) of Optype
         (Einsum) bound to different types").
      2. Making the cast explicit (pos -> double before the einsum) fixes
         that, but onnxruntime's CPUExecutionProvider has no registered
         kernel for double-precision `Cos`/`Sin` ("Could not find an
         implementation for Cos(7)").
    Simplest fix: drop the double-precision detour and do the whole
    sin/cos table in float32. These are bounded (sin/cos of position
    indices scaled by a frequency table, never far from [-1, 1]) --
    float32's ~7 decimal digits is already far more precision than a
    positional embedding needs, so this is not expected to move outputs
    meaningfully; verified numerically against the PyTorch eager
    reference in verify_onnx.py either way.
    """
    import vggttt.nets.vggt.heads.utils as head_utils

    original = head_utils.make_sincos_pos_embed

    def patched(embed_dim, pos, omega_0=100, dtype=torch.float32):
        assert embed_dim % 2 == 0
        omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
        omega = omega / (embed_dim / 2.0)
        omega = 1.0 / omega_0**omega
        pos = pos.reshape(-1).to(torch.float32)
        out = torch.einsum("m,d->md", pos, omega)
        emb_sin = torch.sin(out).to(dtype)
        emb_cos = torch.cos(out).to(dtype)
        return torch.cat([emb_sin, emb_cos], dim=1)

    head_utils.make_sincos_pos_embed = patched
    return original


class VGGTONNXWrapper(nn.Module):
    def __init__(self, model: VGGT, num_ttt_steps: int = 2):
        super().__init__()
        self.model = model
        all_tokens_same_op_order = [
            *[TTTOperator(start=0, end=None, compute_grad=True, update=True, apply=False)] * num_ttt_steps,
            TTTOperator(start=0, end=None, compute_grad=False, update=False, apply=True),
        ]
        self.attn_kwargs = {
            "info": {"ttt_op_order": all_tokens_same_op_order},
            "chunk_size": None,
            "track_details": False,
            "offload_to_cpu": False,
        }

    def forward(self, images: torch.Tensor):
        # images: (S, 3, H, W) float32 in [0, 1] -- matches load_and_preprocess_images' output
        images = images[None]  # add batch dim -> (1, S, 3, H, W), B is always 1 here
        _, _, _, H, W = images.shape

        aggregated_tokens_list, ps_idx, _ = self.model.aggregator(
            images,
            attn_kwargs=self.attn_kwargs,
            add_first_view_token=True,
            memory_efficient_inference=False,
        )
        cam_tokens = aggregated_tokens_list[-1][:, :, 0]  # (1, S, C)
        pose_enc = self.model.camera_head(cam_tokens)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, image_size_hw=(H, W))

        depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images, ps_idx, frames_chunk_size=9999)
        point_map, point_conf = self.model.point_head(aggregated_tokens_list, images, ps_idx, frames_chunk_size=9999)

        pose = closed_form_inverse_se3(extrinsic.squeeze(0))[None]  # (1, S, 4, 4)

        return pose, intrinsic, point_map, point_conf, depth_map, depth_conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="nvidia/vgg-ttt")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--images", nargs="+",
        default=[
            "data/nerf_real_360/pinecone/images_8/IMG_7238.png",
            "data/nerf_real_360/pinecone/images_8/IMG_7239.png",
        ],
    )
    parser.add_argument("--output", default="vggttt_jax/vggttt_pt.onnx")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--dynamo", action="store_true")
    parser.add_argument(
        "--fp16", action="store_true",
        help="Export directly from a model.half()'d PyTorch model, rather than "
             "post-hoc converting an fp32 ONNX graph -- onnxconverter_common's "
             "and onnxruntime.transformers' float32->float16 converters both "
             "silently produced an empty (0-node) graph for this 5.1GB model.",
    )
    args = parser.parse_args()

    _patch_make_sincos_pos_embed()

    print(f"Loading PyTorch model from {args.checkpoint}...")
    model = VGGT.from_pretrained(args.checkpoint)
    model = model.to(args.device).eval()
    if args.fp16:
        print("Converting model to half precision (model.half())...")
        model = model.half()

    print(f"Preprocessing images: {args.images}")
    images = load_and_preprocess_images(args.images).to(args.device)
    if args.fp16:
        images = images.half()
    print(f"input shape: {tuple(images.shape)} dtype: {images.dtype}")

    wrapper = VGGTONNXWrapper(model, num_ttt_steps=2).to(args.device).eval()

    print("Running eager forward pass (sanity check before export)...")
    t0 = time.time()
    with torch.no_grad():
        eager_out = wrapper(images)
    torch.cuda.synchronize() if args.device == "cuda" else None
    print(f"Eager forward: {time.time() - t0:.1f}s")
    for name, t in zip(OUTPUT_NAMES, eager_out):
        print(f"  {name:12s}: {tuple(t.shape)} dtype={t.dtype}")

    print(f"\nExporting to ONNX (opset {args.opset})...")
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (images,),
            args.output,
            input_names=["images"],
            output_names=OUTPUT_NAMES,
            opset_version=args.opset,
            dynamo=args.dynamo,
        )
    print(f"Export took {time.time() - t0:.1f}s -> {args.output}")


if __name__ == "__main__":
    main()
