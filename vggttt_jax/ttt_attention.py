import math
import jax
import jax.numpy as jnp
from flax import linen as nn
from .ttt import ttt_apply, inv_softplus

class ShortConv(nn.Module):
    dim: int
    kernel_size: int

    @nn.compact
    def __call__(self, x, patch_h, patch_w, num_suffix_tokens, num_prefix_tokens):
        # x: [b, num_heads, num_tokens, d]
        b, num_heads, num_tokens, d = x.shape
        num_img_tokens = patch_h * patch_w

        # Suffix tokens are per-sequence
        x_main = x[:, :, :num_tokens - num_suffix_tokens, :]
        suffix = x[:, :, num_tokens - num_suffix_tokens:, :]

        # Prefix tokens are per-image
        num_per_img_tokens = num_img_tokens + num_prefix_tokens
        n = x_main.shape[2] // num_per_img_tokens
        
        # Reshape to [b, num_heads, n, t, d]
        x_main = x_main.reshape(b, num_heads, n, num_per_img_tokens, d)
        prefix = x_main[:, :, :, :num_prefix_tokens, :]
        img_tokens = x_main[:, :, :, num_prefix_tokens:, :]

        # Rearrange image tokens to NHWC for JAX Conv: [b * n, h, w, num_heads * d]
        img_tokens = jnp.transpose(img_tokens, (0, 2, 3, 1, 4)) # [b, n, h*w, num_heads, d]
        img_tokens = img_tokens.reshape(b * n, patch_h, patch_w, num_heads * d)

        # Depthwise 2D convolution in JAX using feature_group_count
        # Named "conf" to match PyTorch's state_dict parameter names
        conv_out = nn.Conv(
            features=self.dim,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding="SAME",
            feature_group_count=self.dim,
            use_bias=False,
            name="conf"
        )(img_tokens)

        # Reshape back: [b * n, h, w, num_heads * d] -> [b, num_heads, n, h*w, d]
        conv_out = conv_out.reshape(b, n, patch_h, patch_w, num_heads, d)
        conv_out = conv_out.reshape(b, n, patch_h * patch_w, num_heads, d)
        conv_out = jnp.transpose(conv_out, (0, 3, 1, 2, 4)) # [b, num_heads, n, h*w, d]

        # Concat prefix back
        x_main = jnp.concatenate([prefix, conv_out], axis=3) # [b, num_heads, n, num_per_img_tokens, d]
        x_main = x_main.reshape(b, num_heads, n * num_per_img_tokens, d)

        # Concat suffix back
        output = jnp.concatenate([x_main, suffix], axis=2)
        return output

class ShortConvsModule(nn.Module):
    dim: int
    short_conv_size_qkv: tuple

    @nn.compact
    def __call__(self, x, i, patch_h, patch_w, num_suffix_tokens, num_prefix_tokens):
        kernel_size = self.short_conv_size_qkv[i]
        if kernel_size > 0:
            return ShortConv(self.dim, kernel_size, name=f"layers_{i}")(
                x, patch_h, patch_w, num_suffix_tokens, num_prefix_tokens
            )
        else:
            return x

class FastWeightAttention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = True
    proj_bias: bool = True
    qk_norm: bool = False
    mlp_ratio: int = 4
    base_lr: float = 0.01
    muon_update_steps: int = 5
    short_conv_size_qkv: tuple = (0, 0, 3)
    div_lr_by_seq_len: bool = False
    num_steps: int = 2

    def setup(self):
        assert self.dim % self.num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = self.dim // self.num_heads
        
        # Initialize projections
        self.qkv = nn.Dense(features=self.dim * 3, use_bias=self.qkv_bias, name="qkv")
        self.proj = nn.Dense(features=self.dim, use_bias=self.proj_bias, name="proj")
        
        # Fast weight learning rates projector
        self.lr_fc = nn.Dense(features=self.num_heads * 3, name="lr_fc")
        self.base_lr_inv = inv_softplus(self.base_lr)
        
        # Short convolutions
        self.using_short_conv_qkv = any(self.short_conv_size_qkv)
        if self.using_short_conv_qkv:
            self.short_conv_qkv = ShortConvsModule(self.dim, self.short_conv_size_qkv, name="short_conv_qkv")

        # Fast weight parameter buffers (they are loaded from msgpack)
        d_in = d_out = self.head_dim
        d_h = int(self.head_dim * self.mlp_ratio)
        self.w0 = self.param("w0", lambda rng, shape: jnp.zeros(shape), (self.num_heads, d_in, d_h))
        self.w1 = self.param("w1", lambda rng, shape: jnp.zeros(shape), (self.num_heads, d_h, d_out))
        self.w2 = self.param("w2", lambda rng, shape: jnp.zeros(shape), (self.num_heads, d_in, d_h))

    def __call__(self, x, pos=None, pos_max=None, patch_h=None, patch_w=None, num_suffix_tokens=0, num_prefix_tokens=0, rope_fn=None):
        B, N, C = x.shape

        # Repeat weights for batch dimension
        w0_batched = jnp.tile(self.w0, (B, 1, 1))
        w1_batched = jnp.tile(self.w1, (B, 1, 1))
        w2_batched = jnp.tile(self.w2, (B, 1, 1))

        # 2. Compute Q, K, V
        qkv = self.qkv(x) # [B, N, 3 * dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4)) # [3, B, num_heads, N, head_dim]
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply Silu activation (and short conv if active)
        if self.using_short_conv_qkv and N > num_prefix_tokens + num_suffix_tokens:
            q = jax.nn.silu(self.short_conv_qkv(q, 0, patch_h, patch_w, num_suffix_tokens, num_prefix_tokens))
            k = jax.nn.silu(self.short_conv_qkv(k, 1, patch_h, patch_w, num_suffix_tokens, num_prefix_tokens))
            v = jax.nn.silu(self.short_conv_qkv(v, 2, patch_h, patch_w, num_suffix_tokens, num_prefix_tokens))
        else:
            q = jax.nn.silu(q)
            k = jax.nn.silu(k)
            v = jax.nn.silu(v)


        # Apply ROPE
        if rope_fn is not None:
            q = rope_fn(q, pos, pos_max=pos_max)
            k = rope_fn(k, pos, pos_max=pos_max)

        # Reshape for TTT: [B, num_heads, N, head_dim] -> [B * num_heads, N, head_dim]
        q = q.reshape(B * self.num_heads, N, self.head_dim)
        k = k.reshape(B * self.num_heads, N, self.head_dim)
        v = v.reshape(B * self.num_heads, N, self.head_dim)

        # Normalization
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-5)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-5)

        # 3. Compute learning rates
        lr = self.lr_fc(x) # [B, N, num_heads * 3]
        lr = jax.nn.softplus(lr + self.base_lr_inv)
        if self.div_lr_by_seq_len:
            lr = lr / N
            
        lr = lr.reshape(B, N, 3, self.num_heads)
        lr = jnp.expand_dims(lr, axis=-1) # [B, N, 3, num_heads, 1]
        lr = jnp.transpose(lr, (2, 0, 3, 1, 4)) # [3, B, num_heads, N, 1]
        lr = lr.reshape(3, B * self.num_heads, N, 1)
        lr0, lr1, lr2 = lr[0], lr[1], lr[2]

        # 4. Run test-time training optimization
        output = ttt_apply(
            w0_batched, w1_batched, w2_batched,
            q, k, v,
            lr0, lr1, lr2,
            num_steps=self.num_steps,
            muon_steps=self.muon_update_steps
        ) # [B * num_heads, N, head_dim]

        # 5. Rearrange back and project
        output = output.reshape(B, self.num_heads, N, self.head_dim)
        output = jnp.transpose(output, (0, 2, 1, 3)) # [B, N, num_heads, head_dim]
        output = output.reshape(B, N, self.dim)
        
        output = self.proj(output)
        return output
