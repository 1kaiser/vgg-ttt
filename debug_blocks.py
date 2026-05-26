"""
Per-block debugging script: find exactly which block diverges in the aggregator.

Strategy:
1. Run the PyTorch aggregator and capture intermediate block outputs.
2. Run the JAX aggregator and capture intermediate block outputs.
3. Compare layer by layer.
"""
import os, sys, math
import torch
import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
from flax import serialization

sys.path.append("/home/kaiser/projects/vgg-ttt")
from vggttt.nets.vggt.models.vggt import VGGT
from vggttt.nets.vggt.img import load_and_preprocess_images
from vggttt_jax.models import (
    Aggregator as AggregatorJAX, VGGT as VGGTJAX,
    slice_expand_and_flatten_jax, interpolate_pos_encoding
)
from vggttt_jax.ttt_attention import FastWeightAttention

def main():
    print("Loading PyTorch model on CUDA...")
    pt_model = VGGT.from_pretrained("nvidia/vgg-ttt")
    pt_model = pt_model.to("cuda").eval()

    print("Loading JAX weights...")
    weights_path = "/home/kaiser/projects/vgg-ttt/vggttt_jax/vggttt.msgpack"
    with open(weights_path, "rb") as f:
        variables = serialization.msgpack_restore(f.read())
    jax_params = variables["params"]

    img_paths = [
        "data/nerf_real_360/pinecone/images_8/IMG_7238.png",
        "data/nerf_real_360/pinecone/images_8/IMG_7239.png"
    ]
    print("Preprocessing images...")
    images_pt = load_and_preprocess_images(img_paths)
    images_pt_batched = images_pt[None].to("cuda")
    images_jax = jnp.expand_dims(images_pt.permute(0, 2, 3, 1).numpy(), axis=0)

    B, S, C, H, W = images_pt_batched.shape
    patch_size = 14

    # ---- PyTorch: obtain intermediate tokens after PatchEmbed ----
    print("\n--- PyTorch PatchEmbed + token construction ---")
    with torch.no_grad():
        mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1,1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1,1,3,1,1)
        imgs_norm = (images_pt_batched - mean) / std
        imgs_flat = imgs_norm.view(B*S, C, H, W)
        pe_out = pt_model.aggregator.patch_embed(imgs_flat)
        pt_patch_tokens = pe_out["x_norm_patchtokens"]          # [B*S, P, C]

        # camera + register tokens
        from vggttt.nets.vggt.models.aggregator import slice_expand_and_flatten
        pt_cam_tok = pt_model.aggregator.camera_token            # [1, 2, 1, 1024]
        pt_reg_tok = pt_model.aggregator.register_token          # [1, 2, 4, 1024]
        cam_exp = slice_expand_and_flatten(pt_cam_tok, B, S, True)
        reg_exp = slice_expand_and_flatten(pt_reg_tok, B, S, True)

        pt_tokens = torch.cat([cam_exp, reg_exp, pt_patch_tokens], dim=1)  # [B*S, 1041, 1024]
        print("PT initial tokens shape:", pt_tokens.shape, "max abs:", pt_tokens.abs().max().item())

    # ---- JAX: obtain initial tokens after PatchEmbed ----
    print("\n--- JAX PatchEmbed + token construction ---")
    agg_params = jax_params["aggregator"]
    mean_j = jnp.array([0.485,0.456,0.406]).reshape(1,1,1,1,3)
    std_j  = jnp.array([0.229,0.224,0.225]).reshape(1,1,1,1,3)
    imgs_norm_j = (images_jax - mean_j) / std_j
    imgs_flat_j = imgs_norm_j.reshape(B*S, H, W, C)

    from vggttt_jax.models import DinoVisionTransformer as DinoViT
    patch_embed_jax = DinoViT(
        img_size=518, patch_size=14, embed_dim=1024, depth=24,
        num_heads=16, mlp_ratio=4.0, num_register_tokens=4,
        qkv_bias=True, ffn_bias=True, proj_bias=True, init_values=1.0,
        interpolate_antialias=True
    )
    pe_vars = {"params": agg_params["patch_embed"]}
    jax_patch_out = patch_embed_jax.apply(pe_vars, imgs_flat_j)
    jax_patch_tokens = jax_patch_out["x_norm_patchtokens"]       # [B*S, P, 1024]

    jax_cam_tok = agg_params["camera_token"]   # [1, 2, 1, 1024]
    jax_reg_tok = agg_params["register_token"] # [1, 2, 4, 1024]
    jax_cam_exp = slice_expand_and_flatten_jax(jax_cam_tok, B, S, True)  # [B*S, 1, 1024]
    jax_reg_exp = slice_expand_and_flatten_jax(jax_reg_tok, B, S, True)  # [B*S, 4, 1024]
    jax_tokens  = jnp.concatenate([jax_cam_exp, jax_reg_exp, jax_patch_tokens], axis=1) # [B*S, 1041, 1024]
    print("JAX initial tokens shape:", jax_tokens.shape, "max abs:", np.max(np.abs(jax_tokens)))

    diff_initial = np.max(np.abs(pt_tokens.detach().cpu().numpy() - np.array(jax_tokens)))
    print(f"Initial tokens Max Abs Diff: {diff_initial:.6e}")

    # ---- Per-block comparison: frame_blocks then global_blocks alternating ----
    print("\n--- Per-block comparison (first 6 pairs) ---")
    # PyTorch uses alternating frame/global, one block per layer
    # JAX uses the same: frame_blocks[i] then global_blocks[i]
    with torch.no_grad():
        pt_x = pt_tokens.clone()
        jax_x = jax_tokens

        from vggttt_jax.models import Block as BlockJAX
        from vggttt_jax.ttt_attention import FastWeightAttention
        from vggttt_jax.rope import RotaryPositionEmbedding2D, PositionGetter

        rope_jax = RotaryPositionEmbedding2D(frequency=100.0)
        pos_getter = PositionGetter()

        patch_h, patch_w = H // patch_size, W // patch_size
        pos = pos_getter(B*S, patch_h, patch_w)
        pos_max = max(patch_h, patch_w) + 1
        pos = pos + 1
        pos_special = jnp.zeros((B*S, 5, 2), dtype=pos.dtype)
        pos = jnp.concatenate([pos_special, pos], axis=1)

        def rope_fn_jax(x, pos, pos_max=None):
            return rope_jax(x, pos, pos_max=pos_max)

        pos_pt = torch.from_numpy(np.array(pos)).to("cuda")

        for block_idx in range(min(6, 24)):
            # PyTorch frame block inputs
            pt_x_input = pt_x.clone()
            
            # Reset JAX frame input to match PyTorch exactly
            jax_x_input = jnp.array(pt_x_input.detach().cpu().numpy())

            # PyTorch frame block
            pt_frame_block = pt_model.aggregator.frame_blocks[block_idx]
            pt_global_block = pt_model.aggregator.global_blocks[block_idx]

            # JAX frame block
            frame_block_params = {"params": agg_params["frame_blocks"][f"layers_{block_idx}"]}
            global_block_params = {"params": agg_params["global_blocks"][f"layers_{block_idx}"]}

            # Frame block pass
            pt_x_after_frame = pt_frame_block(pt_x_input, pos=pos_pt, pos_max=pos_max)
            frame_block_jax = BlockJAX(
                dim=1024, num_heads=16, mlp_ratio=4.0,
                qkv_bias=True, proj_bias=True, ffn_bias=True,
                init_values=0.01, qk_norm=True
            )
            jax_x_after_frame = frame_block_jax.apply(
                frame_block_params, jax_x_input, pos=pos, pos_max=pos_max, rope_fn=rope_fn_jax
            )

            diff_frame = np.max(np.abs(
                pt_x_after_frame.detach().cpu().numpy() - np.array(jax_x_after_frame)
            ))
            print(f"Block {block_idx} FRAME diff (isolated): {diff_frame:.4e}")

            # Global block pass
            N_tok = pt_x_after_frame.shape[1]
            pt_x_glob = pt_x_after_frame.view(B, S * N_tok, 1024)
            pos_global_pt = pos_pt.view(B, S * N_tok, 2)
            
            # Reset JAX global input to match PyTorch exactly
            jax_x_glob_input = jnp.array(pt_x_glob.detach().cpu().numpy())
            pos_global = pos.reshape(B, S * N_tok, 2)

            # PyTorch global block
            pt_x_glob_out = pt_global_block(
                pt_x_glob, pos=pos_global_pt, pos_max=pos_max,
                patch_h=patch_h, patch_w=patch_w, num_prefix_tokens=5
            )
            pt_x = pt_x_glob_out.reshape(B*S, N_tok, 1024)

            # JAX global block
            global_blk_jax = BlockJAX(
                dim=1024, num_heads=16, mlp_ratio=4.0,
                qkv_bias=True, proj_bias=True, ffn_bias=True,
                init_values=0.01, qk_norm=True,
                attn_class=FastWeightAttention
            )
            jax_x_glob_out = global_blk_jax.apply(
                global_block_params, jax_x_glob_input,
                pos=pos_global, pos_max=pos_max, patch_h=patch_h, patch_w=patch_w,
                num_prefix_tokens=5, rope_fn=rope_fn_jax
            )
            jax_x = jax_x_glob_out.reshape(B*S, N_tok, 1024)

            diff_global = np.max(np.abs(
                pt_x.detach().cpu().numpy() - np.array(jax_x)
            ))
            print(f"Block {block_idx} GLOBAL diff (isolated): {diff_global:.4e}")

if __name__ == "__main__":
    main()
