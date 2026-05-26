import os
import sys
import torch
import numpy as np

# Force CPU backend for JAX
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
from flax import serialization

sys.path.append("/home/kaiser/projects/vgg-ttt")

from vggttt.nets.vggt.models.vggt import VGGT
from vggttt.nets.vggt.img import load_and_preprocess_images
from vggttt_jax.models import VGGT as VGGTJAX

def main():
    print("Loading PyTorch reference model on CUDA...")
    pt_model = VGGT.from_pretrained("nvidia/vgg-ttt")
    pt_model = pt_model.to("cuda").eval()

    print("Loading JAX model weights...")
    weights_path = "/home/kaiser/projects/vgg-ttt/vggttt_jax/vggttt.msgpack"
    with open(weights_path, "rb") as f:
        serialized_bytes = f.read()
    variables = serialization.msgpack_restore(serialized_bytes)

    jax_model = VGGTJAX()

    # Preprocess test images
    img_paths = [
        "data/nerf_real_360/pinecone/images_8/IMG_7238.png",
        "data/nerf_real_360/pinecone/images_8/IMG_7239.png"
    ]
    print("Preprocessing images via PyTorch...")
    images_pt = load_and_preprocess_images(img_paths)  # [2, 3, 392, 518]
    images_pt_batched = images_pt[None].to("cuda")  # [1, 2, 3, 392, 518] on CUDA
    
    # JAX inputs [1, 2, 392, 518, 3] on CPU
    images_jax = jnp.expand_dims(images_pt.permute(0, 2, 3, 1).numpy(), axis=0)

    # Print PyTorch interpolate configuration
    print("PyTorch patch_embed.interpolate_antialias:", pt_model.aggregator.patch_embed.interpolate_antialias)
    print("PyTorch patch_embed.interpolate_offset:", pt_model.aggregator.patch_embed.interpolate_offset)

    print("--- 1. Comparing PatchEmbed (DinoVisionTransformer) ---")
    with torch.no_grad():
        # PyTorch normalizes images in aggregator before patch_embed
        mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 1, 3, 1, 1)
        normalized_images_pt = (images_pt_batched - mean) / std
        B, S, C, H, W = normalized_images_pt.shape
        normalized_images_pt_flat = normalized_images_pt.view(B * S, C, H, W)
        
        # PatchEmbed forward
        pt_patch_out_dict = pt_model.aggregator.patch_embed(normalized_images_pt_flat)
        pt_patch_tokens = pt_patch_out_dict["x_norm_patchtokens"]  # [B*S, P, C]

    # JAX PatchEmbed
    jax_params = variables["params"]
    aggregator_variables = {"params": jax_params["aggregator"]}
    
    # Normalizing JAX input
    mean_jax = jnp.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 1, 3)
    std_jax = jnp.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 1, 3)
    normalized_images_jax = (images_jax - mean_jax) / std_jax
    normalized_images_jax_flat = normalized_images_jax.reshape(B * S, H, W, C)
    
    # Apply JAX patch_embed with interpolate_antialias=True
    from vggttt_jax.models import DinoVisionTransformer as DinoVisionTransformerJAX
    patch_embed_jax_aa = DinoVisionTransformerJAX(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        init_values=1.0,
        interpolate_antialias=True
    )
    # Apply JAX patch_embed with interpolate_antialias=False
    patch_embed_jax_noaa = DinoVisionTransformerJAX(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        init_values=1.0,
        interpolate_antialias=False
    )
    
    pe_vars = {"params": jax_params["aggregator"]["patch_embed"]}
    
    # Compare both JAX versions to PyTorch
    jax_patch_out_aa = patch_embed_jax_aa.apply(pe_vars, normalized_images_jax_flat)
    jax_patch_tokens_aa = jax_patch_out_aa["x_norm_patchtokens"]
    
    jax_patch_out_noaa = patch_embed_jax_noaa.apply(pe_vars, normalized_images_jax_flat)
    jax_patch_tokens_noaa = jax_patch_out_noaa["x_norm_patchtokens"]

    diff_patch_aa = np.max(np.abs(pt_patch_tokens.cpu().numpy() - np.array(jax_patch_tokens_aa)))
    diff_patch_noaa = np.max(np.abs(pt_patch_tokens.cpu().numpy() - np.array(jax_patch_tokens_noaa)))
    print(f"PatchEmbed tokens shape PT: {pt_patch_tokens.shape}, JAX: {jax_patch_tokens_aa.shape}")
    print(f"PatchEmbed Max Abs Diff with antialias=True:  {diff_patch_aa:.6e}")
    print(f"PatchEmbed Max Abs Diff with antialias=False: {diff_patch_noaa:.6e}")

    # Use the best one (aa) for subsequent steps
    jax_patch_tokens = jax_patch_tokens_aa
    
    print("--- 2. Comparing Initial Tokens (Pre-Blocks) ---")
    with torch.no_grad():
        pt_camera_token = pt_model.aggregator.camera_token
        pt_register_token = pt_model.aggregator.register_token
        
    jax_camera_token = jax_params["aggregator"]["camera_token"]
    jax_register_token = jax_params["aggregator"]["register_token"]
    
    diff_cam_tok = np.max(np.abs(pt_camera_token.detach().cpu().numpy() - jax_camera_token))
    diff_reg_tok = np.max(np.abs(pt_register_token.detach().cpu().numpy() - jax_register_token))
    print(f"Camera token Max Abs Diff: {diff_cam_tok:.6e}")
    print(f"Register token Max Abs Diff: {diff_reg_tok:.6e}")

    print("--- 3. Comparing Full Aggregator Outputs ---")
    with torch.no_grad():
        pt_agg_out_list, pt_ps_idx, _ = pt_model.aggregator(images_pt_batched)
        pt_agg_last = pt_agg_out_list[-1]  # [B, S, N_tokens, C]

    # Run JAX aggregator
    from vggttt_jax.models import Aggregator as AggregatorJAX
    agg_jax = AggregatorJAX(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        patch_embed_name="dinov2_vitl14_reg",
        rope_freq=100.0
    )
    jax_agg_out_list, jax_ps_idx, _ = agg_jax.apply(aggregator_variables, images_jax)
    jax_agg_last = jax_agg_out_list[-1]

    diff_agg = np.max(np.abs(pt_agg_last.detach().cpu().numpy() - np.array(jax_agg_last)))
    print(f"Aggregator output shape PT: {pt_agg_last.shape}, JAX: {jax_agg_last.shape}")
    print(f"Aggregator Last Layer Max Abs Diff: {diff_agg:.6e}")

    print("--- 4. Comparing Camera Head ---")
    pt_cam_tokens = pt_agg_last[:, :, 0]
    jax_cam_tokens = jax_agg_last[:, :, 0]
    
    with torch.no_grad():
        pt_pose_enc_list = pt_model.camera_head(pt_cam_tokens)
        pt_pose_enc = pt_pose_enc_list[-1]

    from vggttt_jax.models import CameraHead as CameraHeadJAX
    cam_head_jax = CameraHeadJAX(
        dim_in=2048
    )
    cam_vars = {"params": jax_params["camera_head"]}
    jax_pose_enc_list = cam_head_jax.apply(cam_vars, jax_cam_tokens)
    jax_pose_enc = jax_pose_enc_list[-1]

    diff_pose_enc = np.max(np.abs(pt_pose_enc.detach().cpu().numpy() - np.array(jax_pose_enc)))
    print(f"CameraHead output shape PT: {pt_pose_enc.shape}, JAX: {jax_pose_enc.shape}")
    print(f"CameraHead Pose Enc Max Abs Diff: {diff_pose_enc:.6e}")

    print("--- 5. Comparing Decoded Extrinsics & Intrinsics ---")
    from vggttt.nets.vggt.utils.pose_enc import pose_encoding_to_extri_intri
    with torch.no_grad():
        pt_extrinsic, pt_intrinsic = pose_encoding_to_extri_intri(pt_pose_enc, image_size_hw=(H, W))
        pt_pose = pt_model.infer(images_pt.to("cuda"), num_ttt_steps=2)["pose"]

    # Decode JAX pose
    T_jax = jax_pose_enc[..., :3]
    quat_jax = jax_pose_enc[..., 3:7]
    fov_h_jax = jax_pose_enc[..., 7]
    fov_w_jax = jax_pose_enc[..., 8]

    from vggttt_jax.models import quat_to_mat_jax, closed_form_inverse_se3_jax
    R_jax = quat_to_mat_jax(quat_jax)
    extrinsics_jax = jnp.concatenate([R_jax, jnp.expand_dims(T_jax, -1)], axis=-1)
    pose_jax = closed_form_inverse_se3_jax(extrinsics_jax)

    fy_jax = (H / 2.0) / jnp.tan(fov_h_jax / 2.0)
    fx_jax = (W / 2.0) / jnp.tan(fov_w_jax / 2.0)
    
    intrinsics_jax = jnp.zeros(jax_pose_enc.shape[:2] + (3, 3))
    intrinsics_jax = intrinsics_jax.at[..., 0, 0].set(fx_jax)
    intrinsics_jax = intrinsics_jax.at[..., 1, 1].set(fy_jax)
    intrinsics_jax = intrinsics_jax.at[..., 0, 2].set(W / 2.0)
    intrinsics_jax = intrinsics_jax.at[..., 1, 2].set(H / 2.0)
    intrinsics_jax = intrinsics_jax.at[..., 2, 2].set(1.0)

    diff_ext = np.max(np.abs(pt_extrinsic.detach().cpu().numpy() - np.array(extrinsics_jax[0])))
    diff_int = np.max(np.abs(pt_intrinsic.detach().cpu().numpy() - np.array(intrinsics_jax[0])))
    diff_pose_final = np.max(np.abs(pt_pose.detach().cpu().numpy() - np.array(pose_jax[0])))
    
    print(f"Extrinsic Max Abs Diff: {diff_ext:.6e}")
    print(f"Intrinsic Max Abs Diff: {diff_int:.6e}")
    print(f"Final Pose Max Abs Diff: {diff_pose_final:.6e}")

    print("--- 6. Comparing Depth Head ---")
    with torch.no_grad():
        pt_depth_map, pt_depth_conf = pt_model.depth_head(pt_agg_out_list, images_pt_batched, pt_ps_idx)

    # JAX Depth Head
    from vggttt_jax.models import DPTHead as DPTHeadJAX
    depth_head_jax = DPTHeadJAX(
        dim_in=2048,
        output_dim=2,
        activation="exp",
        conf_activation="expp1"
    )
    depth_vars = {"params": jax_params["depth_head"]}
    jax_depth_map, jax_depth_conf = depth_head_jax.apply(depth_vars, jax_agg_out_list, images_jax, jax_ps_idx)

    diff_depth = np.max(np.abs(pt_depth_map.detach().cpu().numpy() - np.array(jax_depth_map)))
    diff_depth_conf = np.max(np.abs(pt_depth_conf.detach().cpu().numpy() - np.array(jax_depth_conf)))
    print(f"Depth Map Max Abs Diff: {diff_depth:.6e}")
    print(f"Depth Conf Max Abs Diff: {diff_depth_conf:.6e}")

if __name__ == "__main__":
    main()
