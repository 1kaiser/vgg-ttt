import jax
import jax.numpy as jnp

class PositionGetter:
    def __call__(self, batch_size: int, height: int, width: int):
        y_coords = jnp.arange(height)
        x_coords = jnp.arange(width)
        # Generate grid positions
        grid_y, grid_x = jnp.meshgrid(y_coords, x_coords, indexing="ij")
        positions = jnp.stack([grid_y.ravel(), grid_x.ravel()], axis=-1) # [H*W, 2]
        return jnp.tile(jnp.expand_dims(positions, 0), (batch_size, 1, 1))

class RotaryPositionEmbedding2D:
    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor

    def _compute_frequency_components(self, dim: int, seq_len: int, dtype):
        exponents = jnp.arange(0, dim, 2, dtype=jnp.float32) / dim
        inv_freq = 1.0 / (self.base_frequency ** exponents)
        positions = jnp.arange(seq_len, dtype=jnp.float32)
        angles = jnp.outer(positions, inv_freq) # [seq_len, dim//2]
        
        angles = angles.astype(dtype)
        angles = jnp.concatenate([angles, angles], axis=-1)
        cos_components = jnp.cos(angles)
        sin_components = jnp.sin(angles)
        return cos_components, sin_components

    def _rotate_features(self, x):
        feature_dim = x.shape[-1]
        half = feature_dim // 2
        x1, x2 = x[..., :half], x[..., half:]
        return jnp.concatenate([-x2, x1], axis=-1)

    def _apply_1d_rope(self, tokens, positions, cos_comp, sin_comp):
        # cos_comp: [pos_max, dim], positions: [B, N]
        # output shape: [B, 1, N, dim]
        cos = cos_comp[positions][:, None, :, :]
        sin = sin_comp[positions][:, None, :, :]
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def __call__(self, tokens, positions, pos_max=None):
        # tokens: [B, n_heads, N, head_dim]
        # positions: [B, N, 2] (y, x coordinates)
        head_dim = tokens.shape[-1]
        assert head_dim % 2 == 0, "Feature dimension must be even"
        
        feature_dim = head_dim // 2
        if pos_max is None:
            pos_max = int(jnp.max(positions)) + 1
            
        cos_comp, sin_comp = self._compute_frequency_components(feature_dim, pos_max, tokens.dtype)
        
        # Split tokens for vertical and horizontal processing
        half_dim = tokens.shape[-1] // 2
        vertical_features = tokens[..., :half_dim]
        horizontal_features = tokens[..., half_dim:]
        
        # Apply RoPE separately for each dimension
        vertical_features = self._apply_1d_rope(vertical_features, positions[..., 0], cos_comp, sin_comp)
        horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 1], cos_comp, sin_comp)
        
        return jnp.concatenate([vertical_features, horizontal_features], axis=-1)
