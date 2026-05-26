import math
import jax
import jax.numpy as jnp
from flax import linen as nn
from .ttt_attention import FastWeightAttention, ShortConv
from .rope import RotaryPositionEmbedding2D, PositionGetter

# OpenCV resnet mean and std constants
_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

def transpose_conv2d(w):
    return w.transpose(2, 3, 1, 0)

def interpolate_pos_encoding(pos_embed, patch_h, patch_w, patch_size, interpolate_antialias=True):
    # pos_embed shape: [1, N + 1, dim]
    dim = pos_embed.shape[-1]
    class_pos_embed = pos_embed[:, :1]
    patch_pos_embed = pos_embed[:, 1:]
    
    # N is the number of patches in default pos_embed
    N = patch_pos_embed.shape[1]
    M = int(math.sqrt(N)) # Default grid size (e.g. 37 for 518/14)
    assert N == M * M
    
    # Reshape to [M, M, dim]
    patch_pos_embed = patch_pos_embed.reshape(M, M, dim)
    
    # Resize to [patch_h, patch_w, dim]. antialias=True matches PyTorch default.
    patch_pos_embed = jax.image.resize(
        patch_pos_embed,
        (patch_h, patch_w, dim),
        method="bicubic",
        antialias=interpolate_antialias
    )
    patch_pos_embed = patch_pos_embed.reshape(1, patch_h * patch_w, dim)
    
    return jnp.concatenate([class_pos_embed, patch_pos_embed], axis=1)


def slice_expand_and_flatten_jax(token_tensor, B, S, add_first_view_token=True):
    # token_tensor: [1, 2, X, C]
    X = token_tensor.shape[2]
    C = token_tensor.shape[3]
    if add_first_view_token:
        query = jnp.repeat(token_tensor[:, 0:1, ...], B, axis=0) # [B, 1, X, C]
        others = jnp.repeat(token_tensor[:, 1:2, ...], B, axis=0) # [B, 1, X, C]
        others = jnp.repeat(others, S - 1, axis=1) # [B, S-1, X, C]
        combined = jnp.concatenate([query, others], axis=1) # [B, S, X, C]
    else:
        others = jnp.repeat(token_tensor[:, 1:2, ...], B, axis=0) # [B, 1, X, C]
        combined = jnp.repeat(others, S, axis=1) # [B, S, X, C]
        
    return combined.reshape(B * S, X, C)

def quat_to_mat_jax(quaternions):
    # scalar-last format (ijrk)
    i, j, k, r = jnp.split(quaternions, 4, axis=-1)
    i = jnp.squeeze(i, axis=-1)
    j = jnp.squeeze(j, axis=-1)
    k = jnp.squeeze(k, axis=-1)
    r = jnp.squeeze(r, axis=-1)
    
    two_s = 2.0 / jnp.sum(quaternions * quaternions, axis=-1)
    
    o = jnp.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        axis=-1
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def closed_form_inverse_se3_jax(extrinsics):
    # extrinsics shape: [B, S, 3, 4]
    B, S, _, _ = extrinsics.shape
    extrinsics_flat = extrinsics.reshape(B * S, 3, 4)
    
    R = extrinsics_flat[..., :3, :3]
    T = extrinsics_flat[..., :3, 3:] # [B*S, 3, 1]
    
    R_T = jnp.swapaxes(R, -1, -2) # [B*S, 3, 3]
    T_inv = -jnp.matmul(R_T, T) # [B*S, 3, 1]
    
    top_rows = jnp.concatenate([R_T, T_inv], axis=-1) # [B*S, 3, 4]
    bottom_row = jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]), (B * S, 1, 1)) # [B*S, 1, 4]
    inverted_flat = jnp.concatenate([top_rows, bottom_row], axis=-2) # [B*S, 4, 4]
    
    return inverted_flat.reshape(B, S, 4, 4)

def activate_pose_jax(pred_pose_enc):
    T = pred_pose_enc[..., :3]
    quat = pred_pose_enc[..., 3:7]
    fl = pred_pose_enc[..., 7:]
    
    # Default translation and quat are linear; fl (fov) is ReLU activated
    fl = jax.nn.relu(fl)
    return jnp.concatenate([T, quat, fl], axis=-1)

def resize_with_align_corners(image, output_shape, method="linear"):
    # image shape: [B, H, W, C]
    # output_shape: (B, h, w, C)
    input_shape = image.shape
    spatial_dims = (1, 2)
    scales = []
    translations = []
    for i in spatial_dims:
        m = input_shape[i]
        n = output_shape[i]
        scale = (n - 1) / (m - 1)
        translation = 0.5 * (1 - scale)
        scales.append(scale)
        translations.append(translation)
    return jax.image.scale_and_translate(
        image, 
        shape=output_shape, 
        spatial_dims=spatial_dims, 
        scale=jnp.array(scales), 
        translation=jnp.array(translations), 
        method=method
    )

class LayerScale(nn.Module):
    dim: int
    init_values: float = 1e-5

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", lambda rng, shape: self.init_values * jnp.ones(shape), (self.dim,))
        return x * scale

class Mlp(nn.Module):
    hidden_features: int
    out_features: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_features, use_bias=self.use_bias, name="fc1")(x)
        x = nn.gelu(x, approximate=False)
        x = nn.Dense(features=self.out_features, use_bias=self.use_bias, name="fc2")(x)
        return x

class Attention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = True
    proj_bias: bool = True
    qk_norm: bool = False
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x, pos=None, pos_max=None, rope_fn=None):
        B, N, C = x.shape
        head_dim = self.dim // self.num_heads
        scale = head_dim ** -0.5

        qkv = nn.Dense(features=self.dim * 3, use_bias=self.qkv_bias, name="qkv")(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4)) # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            q = nn.LayerNorm(use_bias=self.qkv_bias, epsilon=self.epsilon, name="q_norm")(q)
            k = nn.LayerNorm(use_bias=self.qkv_bias, epsilon=self.epsilon, name="k_norm")(k)

        if rope_fn is not None:
            q = rope_fn(q, pos, pos_max=pos_max)
            k = rope_fn(k, pos, pos_max=pos_max)

        # Standard softmax attention
        attn_weights = jnp.matmul(q * scale, jnp.swapaxes(k, -1, -2))
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        x_attn = jnp.matmul(attn_weights, v) # [B, H, N, D]

        # Reshape back to [B, N, C]
        x_attn = jnp.transpose(x_attn, (0, 2, 1, 3))
        x_attn = x_attn.reshape(B, N, C)

        x_attn = nn.Dense(features=self.dim, use_bias=self.proj_bias, name="proj")(x_attn)
        return x_attn

class Block(nn.Module):
    dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    proj_bias: bool = True
    ffn_bias: bool = True
    init_values: float = None
    qk_norm: bool = False
    attn_class: type = Attention
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x, pos=None, pos_max=None, rope_fn=None, **kwargs):
        # Attention block
        norm1 = nn.LayerNorm(epsilon=self.epsilon, name="norm1")(x)
        
        attn_kwargs = {
            "dim": self.dim,
            "num_heads": self.num_heads,
            "qkv_bias": self.qkv_bias,
            "proj_bias": self.proj_bias,
            "qk_norm": self.qk_norm,
            "name": "attn"
        }
        if hasattr(self.attn_class, "__dataclass_fields__") and "epsilon" in self.attn_class.__dataclass_fields__:
            attn_kwargs["epsilon"] = self.epsilon
            
        attn_out = self.attn_class(**attn_kwargs)(norm1, pos=pos, pos_max=pos_max, rope_fn=rope_fn, **kwargs)

        if self.init_values is not None:
            attn_out = LayerScale(dim=self.dim, init_values=self.init_values, name="ls1")(attn_out)
        x = x + attn_out

        # MLP block
        norm2 = nn.LayerNorm(epsilon=self.epsilon, name="norm2")(x)
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        mlp_out = Mlp(
            hidden_features=mlp_hidden_dim,
            out_features=self.dim,
            use_bias=self.ffn_bias,
            name="mlp"
        )(norm2)

        if self.init_values is not None:
            mlp_out = LayerScale(dim=self.dim, init_values=self.init_values, name="ls2")(mlp_out)
        x = x + mlp_out

        return x

class PatchEmbed(nn.Module):
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 768

    @nn.compact
    def __call__(self, x):
        # input is standard NHWC image tensor [B, H, W, 3]
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            name="proj"
        )(x)
        B, H_g, W_g, C_g = x.shape
        x = x.reshape(B, H_g * W_g, C_g)
        return x

class BlocksModule(nn.Module):
    depth: int
    embed_dim: int
    num_heads: int
    mlp_ratio: float
    qkv_bias: bool
    ffn_bias: bool
    proj_bias: bool
    init_values: float
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x):
        for i in range(self.depth):
            x = Block(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                proj_bias=self.proj_bias,
                ffn_bias=self.ffn_bias,
                init_values=self.init_values,
                epsilon=self.epsilon,
                name=f"layers_{i}"
            )(x)
        return x

class DinoVisionTransformer(nn.Module):
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    ffn_bias: bool = True
    proj_bias: bool = True
    init_values: float = None
    num_register_tokens: int = 0
    interpolate_antialias: bool = True

    def setup(self):
        self.cls_token = self.param("cls_token", lambda rng, shape: jnp.zeros(shape), (1, 1, self.embed_dim))
        if self.num_register_tokens > 0:
            self.register_tokens = self.param("register_tokens", lambda rng, shape: jnp.zeros(shape), (1, self.num_register_tokens, self.embed_dim))
        
        M = self.img_size // self.patch_size
        self.pos_embed = self.param("pos_embed", lambda rng, shape: jnp.zeros(shape), (1, M * M + 1, self.embed_dim))
        
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            name="patch_embed"
        )
        
        self.blocks = BlocksModule(
            depth=self.depth,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            ffn_bias=self.ffn_bias,
            proj_bias=self.proj_bias,
            init_values=self.init_values,
            epsilon=1e-6,
            name="blocks"
        )
        
        self.norm = nn.LayerNorm(epsilon=1e-6, name="norm")

    def __call__(self, x):
        B, H, W, _ = x.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Prepends cls_token
        cls_token = jnp.repeat(self.cls_token, B, axis=0)
        x = jnp.concatenate([cls_token, x], axis=1)
        
        # Add interpolated pos encoding
        pos_embed = interpolate_pos_encoding(self.pos_embed, patch_h, patch_w, self.patch_size, self.interpolate_antialias)
        x = x + pos_embed
        
        # Prepends register tokens
        if self.num_register_tokens > 0:
            reg_tokens = jnp.repeat(self.register_tokens, B, axis=0)
            x = jnp.concatenate([x[:, :1], reg_tokens, x[:, 1:]], axis=1)
            
        x = self.blocks(x)
            
        x_norm = self.norm(x)
        
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x
        }

class FrameBlocksModule(nn.Module):
    dim: int
    num_heads: int
    mlp_ratio: float
    qkv_bias: bool
    proj_bias: bool
    ffn_bias: bool
    init_values: float
    qk_norm: bool
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x, pos, pos_max, rope_fn, idx):
        return Block(
            dim=self.dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            proj_bias=self.proj_bias,
            ffn_bias=self.ffn_bias,
            init_values=self.init_values,
            qk_norm=self.qk_norm,
            epsilon=self.epsilon,
            name=f"layers_{idx}"
        )(x, pos=pos, pos_max=pos_max, rope_fn=rope_fn)

class GlobalBlocksModule(nn.Module):
    dim: int
    num_heads: int
    mlp_ratio: float
    qkv_bias: bool
    proj_bias: bool
    ffn_bias: bool
    init_values: float
    qk_norm: bool
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x, pos, pos_max, patch_h, patch_w, num_prefix_tokens, rope_fn, idx):
        return Block(
            dim=self.dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            proj_bias=self.proj_bias,
            ffn_bias=self.ffn_bias,
            init_values=self.init_values,
            qk_norm=self.qk_norm,
            attn_class=FastWeightAttention,
            epsilon=self.epsilon,
            name=f"layers_{idx}"
        )(
            x,
            pos=pos,
            pos_max=pos_max,
            patch_h=patch_h,
            patch_w=patch_w,
            num_prefix_tokens=num_prefix_tokens,
            rope_fn=rope_fn
        )

class Aggregator(nn.Module):
    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_register_tokens: int = 4
    qkv_bias: bool = True
    proj_bias: bool = True
    ffn_bias: bool = True
    patch_embed_name: str = "dinov2_vitl14_reg"
    aa_order: tuple = ("frame", "global")
    qk_norm: bool = True
    rope_freq: float = 100.0
    init_values: float = 0.01

    def setup(self):
        self.patch_start_idx = 1 + self.num_register_tokens
        
        self.camera_token = self.param("camera_token", lambda rng, shape: jnp.zeros(shape), (1, 2, 1, self.embed_dim))
        self.register_token = self.param("register_token", lambda rng, shape: jnp.zeros(shape), (1, 2, self.num_register_tokens, self.embed_dim))
        
        if "vitl14" in self.patch_embed_name:
            self.patch_embed = DinoVisionTransformer(
                img_size=518,
                patch_size=self.patch_size,
                embed_dim=self.embed_dim,
                depth=24,
                num_heads=16,
                mlp_ratio=4.0,
                num_register_tokens=self.num_register_tokens,
                qkv_bias=True,
                ffn_bias=True,
                proj_bias=True,
                init_values=1.0,
                name="patch_embed"
            )
        else:
            raise ValueError(f"Unsupported patch embedder: {self.patch_embed_name}")
            
        self.rope = RotaryPositionEmbedding2D(frequency=self.rope_freq) if self.rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = FrameBlocksModule(
            dim=self.embed_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            proj_bias=self.proj_bias,
            ffn_bias=self.ffn_bias,
            init_values=self.init_values,
            qk_norm=self.qk_norm,
            name="frame_blocks"
        )
        
        self.global_blocks = GlobalBlocksModule(
            dim=self.embed_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            proj_bias=self.proj_bias,
            ffn_bias=self.ffn_bias,
            init_values=self.init_values,
            qk_norm=self.qk_norm,
            name="global_blocks"
        )

    def __call__(self, images, intermediate_layers_to_return=(4, 11, 17, 23), add_first_view_token=True):
        B, S, H, W, C_in = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        
        # Resnet normalisation
        mean = jnp.array(_RESNET_MEAN).reshape(1, 1, 1, 1, 3)
        std = jnp.array(_RESNET_STD).reshape(1, 1, 1, 1, 3)
        images = (images - mean) / std
        
        # Reshape to [B*S, H, W, 3] for DinoViT
        images = images.reshape(B * S, H, W, C_in)
        patch_tokens_dict = self.patch_embed(images)
        patch_tokens = patch_tokens_dict["x_norm_patchtokens"] # [B*S, P, C]
        
        _, P, C = patch_tokens.shape
        
        camera_token = slice_expand_and_flatten_jax(self.camera_token, B, S, add_first_view_token)
        register_token = slice_expand_and_flatten_jax(self.register_token, B, S, add_first_view_token)
        
        tokens = jnp.concatenate([camera_token, register_token, patch_tokens], axis=1)
        
        pos = None
        pos_max = None
        if self.rope is not None:
            pos = self.position_getter(B * S, patch_h, patch_w)
            pos_max = max(patch_h, patch_w) + 1
            if self.patch_start_idx > 0:
                pos = pos + 1
                pos_special = jnp.zeros((B * S, self.patch_start_idx, 2), dtype=pos.dtype)
                pos = jnp.concatenate([pos_special, pos], axis=1)
                
        _, P, C = tokens.shape
        
        output_list = []
        
        rope_fn = lambda t, positions, pos_max=None: self.rope(t, positions, pos_max=pos_max) if self.rope is not None else t
        
        for aa_block_idx in range(self.depth):
            tokens = tokens.reshape(B * S, P, C)
            tokens = self.frame_blocks(tokens, pos, pos_max, rope_fn, aa_block_idx)
            frame_inter = tokens.reshape(B, S, P, C)
            
            tokens = tokens.reshape(B, S * P, C)
            pos_global = pos.reshape(B, S * P, 2)
            
            tokens = self.global_blocks(
                tokens,
                pos=pos_global,
                pos_max=pos_max,
                patch_h=patch_h,
                patch_w=patch_w,
                num_prefix_tokens=self.patch_start_idx,
                rope_fn=rope_fn,
                idx=aa_block_idx
            )
            global_inter = tokens.reshape(B, S, P, C)
            
            if aa_block_idx in intermediate_layers_to_return:
                concat_inter = jnp.concatenate([frame_inter, global_inter], axis=-1)
                output_list.append(concat_inter)
                
        return output_list, self.patch_start_idx, pos

class TrunkModule(nn.Module):
    dim_in: int
    num_heads: int
    mlp_ratio: int
    init_values: float
    trunk_depth: int

    @nn.compact
    def __call__(self, x):
        for i in range(self.trunk_depth):
            x = Block(
                dim=self.dim_in,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                init_values=self.init_values,
                name=f"layers_{i}"
            )(x)
        return x

class PoseLNModulation(nn.Module):
    dim_in: int

    @nn.compact
    def __call__(self, x):
        x = nn.silu(x)
        x = nn.Dense(features=3 * self.dim_in, name="layers_1")(x)
        return x

class CameraHead(nn.Module):
    dim_in: int = 2048
    trunk_depth: int = 4
    num_heads: int = 16
    mlp_ratio: int = 4
    init_values: float = 0.01

    def setup(self):
        self.target_dim = 9
        self.trunk = TrunkModule(
            dim_in=self.dim_in,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            init_values=self.init_values,
            trunk_depth=self.trunk_depth,
            name="trunk"
        )
        self.token_norm = nn.LayerNorm(epsilon=1e-5, name="token_norm")
        self.trunk_norm = nn.LayerNorm(epsilon=1e-5, name="trunk_norm")
        
        self.empty_pose_tokens = self.param("empty_pose_tokens", lambda rng, shape: jnp.zeros(shape), (1, 1, self.target_dim))
        self.embed_pose = nn.Dense(features=self.dim_in, name="embed_pose")
        
        self.poseLN_modulation = PoseLNModulation(self.dim_in, name="poseLN_modulation")
        
        self.adaln_norm = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6)
        self.pose_branch = Mlp(
            hidden_features=self.dim_in // 2,
            out_features=self.target_dim,
            use_bias=True,
            name="pose_branch"
        )

    def _single_iteration(self, module_input, pose_tokens, pred_pose_enc):
        modulation = self.poseLN_modulation(module_input)
        shift_msa, scale_msa, gate_msa = jnp.split(modulation, 3, axis=-1)
        
        normed = self.adaln_norm(pose_tokens)
        pose_tokens_modulated = gate_msa * (normed * (1.0 + scale_msa) + shift_msa) + pose_tokens
        
        pose_tokens_modulated = self.trunk(pose_tokens_modulated)
            
        pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))
        
        if pred_pose_enc is None:
            pred_pose_enc = pred_pose_enc_delta
        else:
            pred_pose_enc = pred_pose_enc + pred_pose_enc_delta
            
        activated_pose = activate_pose_jax(pred_pose_enc)
        return pred_pose_enc, activated_pose

    def __call__(self, camera_tokens, num_iterations=4):
        B, S, C = camera_tokens.shape
        camera_tokens = self.token_norm(camera_tokens)
        
        pred_pose_enc = None
        pred_pose_enc_list = []
        
        for i in range(num_iterations):
            if i == 0:
                empty = jnp.tile(self.empty_pose_tokens, (B, S, 1))
                module_input = self.embed_pose(empty)
            else:
                module_input = self.embed_pose(pred_pose_enc)
                
            pred_pose_enc, activated_pose = self._single_iteration(
                module_input, camera_tokens, pred_pose_enc if i > 0 else None
            )
            pred_pose_enc_list.append(activated_pose)
            
        return pred_pose_enc_list

class ResidualConvUnit(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        out = nn.relu(x)
        out = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME", name="conv1")(out)
        out = nn.relu(out)
        out = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME", name="conv2")(out)
        return out + nn.relu(x)

class FeatureFusionBlock(nn.Module):
    features: int
    has_residual: bool = True

    @nn.compact
    def __call__(self, *xs, size=None):
        output = xs[0]
        if self.has_residual:
            res = ResidualConvUnit(features=self.features, name="resConfUnit1")(xs[1])
            output = output + res
            
        output = ResidualConvUnit(features=self.features, name="resConfUnit2")(output)
        
        if size is None:
            h, w = output.shape[1] * 2, output.shape[2] * 2
        else:
            h, w = size
            
        output = resize_with_align_corners(output, (output.shape[0], h, w, output.shape[3]))
        output = nn.Conv(features=self.features, kernel_size=(1, 1), name="out_conv")(output)
        return output

class ProjectsModule(nn.Module):
    out_channels: tuple

    @nn.compact
    def __call__(self, xs):
        out = []
        for i, oc in enumerate(self.out_channels):
            out.append(nn.Conv(features=oc, kernel_size=(1, 1), name=f"layers_{i}")(xs[i]))
        return out

class ResizeLayersModule(nn.Module):
    out_channels: tuple

    @nn.compact
    def __call__(self, xs):
        x0 = nn.ConvTranspose(features=self.out_channels[0], kernel_size=(4, 4), strides=(4, 4), padding="VALID", name="layers_0")(xs[0])
        x1 = nn.ConvTranspose(features=self.out_channels[1], kernel_size=(2, 2), strides=(2, 2), padding="VALID", name="layers_1")(xs[1])
        x2 = xs[2]
        x3 = nn.Conv(features=self.out_channels[3], kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)), name="layers_3")(xs[3])
        return [x0, x1, x2, x3]

class OutputConv2Module(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME", name="layers_0")(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.output_dim, kernel_size=(1, 1), name="layers_2")(x)
        return x

class ScratchModule(nn.Module):
    out_channels: tuple
    features: int
    output_dim: int

    def setup(self):
        self.layer1_rn = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME", use_bias=False, name="layer1_rn")
        self.layer2_rn = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME", use_bias=False, name="layer2_rn")
        self.layer3_rn = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME", use_bias=False, name="layer3_rn")
        self.layer4_rn = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME", use_bias=False, name="layer4_rn")
        
        self.refinenet4 = FeatureFusionBlock(features=self.features, has_residual=False, name="refinenet4")
        self.refinenet3 = FeatureFusionBlock(features=self.features, has_residual=True, name="refinenet3")
        self.refinenet2 = FeatureFusionBlock(features=self.features, has_residual=True, name="refinenet2")
        self.refinenet1 = FeatureFusionBlock(features=self.features, has_residual=True, name="refinenet1")
        
        self.output_conv1 = nn.Conv(features=self.features // 2, kernel_size=(3, 3), padding="SAME", name="output_conv1")
        self.output_conv2 = OutputConv2Module(self.output_dim, name="output_conv2")

    def forward_fused(self, xs_resized):
        layer_1, layer_2, layer_3, layer_4 = xs_resized
        l1_rn = self.layer1_rn(layer_1)
        l2_rn = self.layer2_rn(layer_2)
        l3_rn = self.layer3_rn(layer_3)
        l4_rn = self.layer4_rn(layer_4)
        
        out = self.refinenet4(l4_rn, size=l3_rn.shape[1:3])
        out = self.refinenet3(out, l3_rn, size=l2_rn.shape[1:3])
        out = self.refinenet2(out, l2_rn, size=l1_rn.shape[1:3])
        out = self.refinenet1(out, l1_rn)
        
        out = self.output_conv1(out)
        return out

    def forward_head(self, x):
        return self.output_conv2(x)

class DPTHead(nn.Module):
    dim_in: int
    patch_size: int = 14
    output_dim: int = 4
    activation: str = "inv_log"
    conf_activation: str = "expp1"
    features: int = 256
    out_channels: tuple = (256, 512, 1024, 1024)
    pos_embed: bool = True
    down_ratio: int = 1

    def setup(self):
        self.norm = nn.LayerNorm(epsilon=1e-5, name="norm")
        self.projects = ProjectsModule(self.out_channels, name="projects")
        self.resize_layers = ResizeLayersModule(self.out_channels, name="resize_layers")
        self.scratch = ScratchModule(self.out_channels, self.features, self.output_dim, name="scratch")

    def _apply_pos_embed(self, x, W, H, ratio=0.1):
        patch_h, patch_w = x.shape[1:3]
        aspect_ratio = float(W) / float(H)
        diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
        span_x = aspect_ratio / diag_factor
        span_y = 1.0 / diag_factor
        
        left_x = -span_x * (patch_w - 1) / patch_w
        right_x = span_x * (patch_w - 1) / patch_w
        top_y = -span_y * (patch_h - 1) / patch_h
        bottom_y = span_y * (patch_h - 1) / patch_h
        
        x_coords = jnp.linspace(left_x, right_x, num=patch_w)
        y_coords = jnp.linspace(top_y, bottom_y, num=patch_h)
        uu, vv = jnp.meshgrid(x_coords, y_coords, indexing="xy")
        pos_grid = jnp.stack((uu, vv), axis=-1)
        
        H_g, W_g, _ = pos_grid.shape
        pos_flat = pos_grid.reshape(-1, 2)
        
        def make_sincos(pos, embed_dim):
            omega = jnp.arange(embed_dim // 2, dtype=jnp.float32) / (embed_dim / 2.0)
            omega = 1.0 / (100.0 ** omega)
            out = jnp.outer(pos, omega)
            return jnp.concatenate([jnp.sin(out), jnp.cos(out)], axis=-1)
            
        emb_x = make_sincos(pos_flat[:, 0], x.shape[-1] // 2)
        emb_y = make_sincos(pos_flat[:, 1], x.shape[-1] // 2)
        emb = jnp.concatenate([emb_x, emb_y], axis=-1)
        pos_embed = emb.reshape(H_g, W_g, x.shape[-1])
        
        pos_embed = pos_embed * ratio
        return x + jnp.expand_dims(pos_embed, 0)

    def __call__(self, tokens_list, images, patch_start_idx):
        B, S, P, C = tokens_list[0].shape
        _, _, H, W, _ = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        
        projected_tokens = []
        for i, tokens in enumerate(tokens_list):
            x = tokens[:, :, patch_start_idx:, :]
            x = x.reshape(B * S, P - patch_start_idx, C)
            x = self.norm(x)
            x = x.reshape(B * S, patch_h, patch_w, C)
            projected_tokens.append(x)
            
        # Project channels
        projected = self.projects(projected_tokens)
        
        # Apply positional embedding to each projected feature map before resizing
        if self.pos_embed:
            projected = [self._apply_pos_embed(item, W, H) for item in projected]
            
        # Resize/Upsample layers
        resized = self.resize_layers(projected)
            
        # Fusion with scratch refinenet/stem layers
        output_conv1 = self.scratch.forward_fused(resized)
        
        # Interpolate fused output to target resolution
        out_h = int(patch_h * self.patch_size / self.down_ratio)
        out_w = int(patch_w * self.patch_size / self.down_ratio)
        out = resize_with_align_corners(output_conv1, (output_conv1.shape[0], out_h, out_w, output_conv1.shape[3]))
        
        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)
            
        out = self.scratch.forward_head(out)
        
        xyz = out[:, :, :, :-1]
        conf = out[:, :, :, -1]
        
        if self.activation == "inv_log":
            pts3d = jnp.sign(xyz) * jnp.expm1(jnp.abs(xyz))
        elif self.activation == "exp":
            pts3d = jnp.exp(xyz)
        elif self.activation == "linear":
            pts3d = xyz
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
            
        if self.conf_activation == "expp1":
            conf_out = 1.0 + jnp.exp(conf)
        else:
            raise ValueError(f"Unsupported conf_activation: {self.conf_activation}")
            
        pts3d = pts3d.reshape(B, S, out_h, out_w, pts3d.shape[-1])
        conf_out = conf_out.reshape(B, S, out_h, out_w)
        return pts3d, conf_out

class VGGT(nn.Module):
    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    depth: int = 24
    intermediate_layer_idx: tuple = (4, 11, 17, 23)
    patch_embed: str = "dinov2_vitl14_reg"
    rope_freq: float = 100.0

    def setup(self):
        self.aggregator = Aggregator(
            img_size=518,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            depth=self.depth,
            patch_embed_name=self.patch_embed,
            rope_freq=self.rope_freq,
            name="aggregator"
        )
        self.camera_head = CameraHead(
            dim_in=2 * self.embed_dim,
            name="camera_head"
        )
        self.point_head = DPTHead(
            dim_in=2 * self.embed_dim,
            output_dim=4,
            activation="inv_log",
            conf_activation="expp1",
            name="point_head"
        )
        self.depth_head = DPTHead(
            dim_in=2 * self.embed_dim,
            output_dim=2,
            activation="exp",
            conf_activation="expp1",
            name="depth_head"
        )

    def __call__(self, images, add_first_view_token=True):
        # input is standard NHWC images tensor [B, S, H, W, 3] in [0, 1]
        B, S, H, W, _ = images.shape
        
        # 1. Run Aggregator alternating frame/global attention blocks
        aggregated_tokens_list, patch_start_idx, pos = self.aggregator(
            images,
            intermediate_layers_to_return=self.intermediate_layer_idx,
            add_first_view_token=add_first_view_token
        )
        
        # 2. Run CameraHead using camera tokens of the final layer
        cam_tokens = aggregated_tokens_list[-1][:, :, 0] # [B, S, C]
        pose_enc_list = self.camera_head(cam_tokens)
        pose_enc = pose_enc_list[-1]
        
        # Unpack absolute translations, rotation quaternions, and focal lengths
        T = pose_enc[..., :3]
        quat = pose_enc[..., 3:7]
        fov_h = pose_enc[..., 7]
        fov_w = pose_enc[..., 8]
        
        R = quat_to_mat_jax(quat)
        extrinsics = jnp.concatenate([R, jnp.expand_dims(T, -1)], axis=-1) # [B, S, 3, 4]
        
        # Compute pinhole camera intrinsics from FOVs
        fy = (H / 2.0) / jnp.tan(fov_h / 2.0)
        fx = (W / 2.0) / jnp.tan(fov_w / 2.0)
        
        intrinsics = jnp.zeros(pose_enc.shape[:2] + (3, 3))
        intrinsics = intrinsics.at[..., 0, 0].set(fx)
        intrinsics = intrinsics.at[..., 1, 1].set(fy)
        intrinsics = intrinsics.at[..., 0, 2].set(W / 2.0)
        intrinsics = intrinsics.at[..., 1, 2].set(H / 2.0)
        intrinsics = intrinsics.at[..., 2, 2].set(1.0)
        
        # 3. Predict Depth Maps
        depth_map, depth_conf = self.depth_head(
            aggregated_tokens_list, images, patch_start_idx
        )
        
        # 4. Predict Point Maps (Global branch)
        point_map, point_conf = self.point_head(
            aggregated_tokens_list, images, patch_start_idx
        )

        return {
            "pose": closed_form_inverse_se3_jax(extrinsics), # Camera-to-world transform [B, S, 4, 4]
            "intrinsics": intrinsics, # Pinhole camera matrix [B, S, 3, 3]
            "pts3d": point_map, # Reconstructed 3D world coords [B, S, H_out, W_out, 3]
            "conf": point_conf, # Reconstruction confidence [B, S, H_out, W_out]
            "depth": depth_map, # Depth maps [B, S, H_out, W_out, 1]
            "depth_conf": depth_conf, # Depth confidence [B, S, H_out, W_out]
        }
