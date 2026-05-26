import jax
import jax.numpy as jnp
import math

def inv_softplus(x: float):
    return x + math.log(-math.expm1(-x))


def zeropower_via_newtonschulz5(G, steps=4):
    """Newton-Schulz orthogonalisation.
    
    Args:
        G: [b, d_in, d_out] input matrices to orthogonalise.
        steps: number of Newton-Schulz iterations.
    """
    if steps < 0:
        return G

    # X starts as copy of G
    X = G
    
    # Transpose if rows > cols
    transposed = False
    if G.shape[1] > G.shape[2]:
        X = jnp.swapaxes(X, 1, 2)
        transposed = True

    # Normalize by matrix norm (norm over axis 1 and 2)
    norm = jnp.linalg.norm(X, axis=(1, 2), keepdims=True)
    X = X / (norm + 1e-7)

    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(steps):
        # A = X @ X.mT
        A = jnp.matmul(X, jnp.swapaxes(X, -1, -2))
        # B = b * A + c * A @ A
        B = b * A + c * jnp.matmul(A, A)
        # X = a * X + B @ X
        X = a * X + jnp.matmul(B, X)

    if transposed:
        X = jnp.swapaxes(X, 1, 2)
    return X

def swish_glu_fwd(x, w0, w1, w2):
    """Forward pass through local MLP.
    
    Args:
        x: [B, L, D_in] query/key tensor
        w0: [B, D_in, D_h] weights 0
        w1: [B, D_h, D_out] weights 1
        w2: [B, D_in, D_h] weights 2
    """
    gate = jax.nn.silu(jnp.matmul(x, w0))
    hidden = jnp.matmul(x, w2)
    return jnp.matmul(gate * hidden, w1)

def manual_grads(w0, w1, w2, k, v, lr0, lr1, lr2):
    """Manually computed VJPs (gradients) for the Swish-GLU MLP.
    
    Args:
        w0: [B, D_in, D_h]
        w1: [B, D_h, D_out]
        w2: [B, D_in, D_h]
        k: [B, L, D_in] keys
        v: [B, L, D_out] values
        lr0, lr1, lr2: [B, L, 1] learning rates
    """
    gate_before_act = jnp.matmul(k, w0)
    hidden_before_mul = jnp.matmul(k, w2)
    gate = jax.nn.silu(gate_before_act)
    hidden = gate * hidden_before_mul
    
    # Backward pass
    dhidden = jnp.matmul(v, jnp.swapaxes(w1, -1, -2))
    dhidden_before_mul = dhidden * gate
    dgate = dhidden * hidden_before_mul
    
    # silu backprop
    sig = jax.nn.sigmoid(gate_before_act)
    dgate_before_act = dgate * sig * (1.0 + gate_before_act * (1.0 - sig))
    
    # Compute gradients (VJP)
    w1_grad = jnp.matmul(jnp.swapaxes(hidden * lr1, -1, -2), v)
    w0_grad = jnp.matmul(jnp.swapaxes(k * lr0, -1, -2), dgate_before_act)
    w2_grad = jnp.matmul(jnp.swapaxes(k * lr2, -1, -2), dhidden_before_mul)
    
    return w0_grad, w1_grad, w2_grad

def ttt_apply(w0, w1, w2, q, k, v, lr0, lr1, lr2, num_steps=1, muon_steps=4):
    """Runs test-time training optimization and returns output for query q.
    
    Args:
        w0, w1, w2: Initial fast weights.
        q: Queries [B, L, D]
        k: Keys [B, L, D]
        v: Values [B, L, D]
        lr0, lr1, lr2: [B, L, 1] learning rates
        num_steps: Number of TTT steps
        muon_steps: Number of Newton-Schulz steps for optimization
    """
    # Keep track of initial weight norms for projection normalization
    w0_norm = jnp.linalg.norm(w0, axis=1, keepdims=True)
    w1_norm = jnp.linalg.norm(w1, axis=1, keepdims=True)
    w2_norm = jnp.linalg.norm(w2, axis=1, keepdims=True)
    
    # We trace python loop as num_steps is static and small
    for step in range(num_steps):
        w0_grad, w1_grad, w2_grad = manual_grads(w0, w1, w2, k, v, lr0, lr1, lr2)
        
        # NOTE: PyTorch default is norm_ttt_grad=False, so we do NOT scale by seq_len.
        # The gradient already accumulates over the full sequence (sum, not mean).
        
        # Update using Newton-Schulz orthogonalization
        w0 = w0 + zeropower_via_newtonschulz5(w0_grad, muon_steps)
        w1 = w1 + zeropower_via_newtonschulz5(w1_grad, muon_steps)
        w2 = w2 + zeropower_via_newtonschulz5(w2_grad, muon_steps)
        
        # Normalize weights back to initial norms
        w0 = w0 / (jnp.linalg.norm(w0, axis=1, keepdims=True) + 1e-5) * w0_norm
        w1 = w1 / (jnp.linalg.norm(w1, axis=1, keepdims=True) + 1e-5) * w1_norm
        w2 = w2 / (jnp.linalg.norm(w2, axis=1, keepdims=True) + 1e-5) * w2_norm
        
    # Apply optimized weights to query q
    return swish_glu_fwd(q, w0, w1, w2)
