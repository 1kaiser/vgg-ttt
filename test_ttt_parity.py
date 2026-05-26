"""
Targeted comparison: test just the TTT core math (not the full attention layer).
Compares JAX ttt_apply vs the PyTorch fast_weight_swish_glu_weight_norm_mini_batch_apply
using the exact same inputs.
"""
import os, sys, math
import torch
import torch.nn.functional as F
import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
from flax import serialization

sys.path.append("/home/kaiser/projects/vgg-ttt")
from vggttt_jax.ttt import ttt_apply, zeropower_via_newtonschulz5, swish_glu_fwd, manual_grads, inv_softplus

# ---- Replicate PyTorch ttt logic manually (on CPU) ----
def pt_zeropower(G, steps):
    """PyTorch zeropower_via_newtonschulz5 on CPU."""
    if steps < 0:
        return G
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.clone()
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = torch.bmm(X, X.mT)
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)
        X = torch.baddbmm(X, B, X, beta=a)
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X

def pt_silu_backprop(dy, x):
    sigma = torch.sigmoid(x)
    return dy * sigma * (1 + x * (1 - sigma))

def pt_ttt_cpu(w0, w1, w2, q, k, v, lr0, lr1, lr2, muon_steps=5, norm_grad=False):
    """
    PyTorch TTT on CPU matching the logic in fast_weight_swish_glu_weight_norm_mini_batch_apply.
    w0: [BH, d, dh], w1: [BH, dh, d], w2: [BH, d, dh]
    q, k, v: [BH, N, d]
    lr0, lr1, lr2: [BH, N, 1]
    """
    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)
    
    N = k.shape[1]
    
    # Compute gradient (one step, all tokens at once)
    gate_before_act = k @ w0
    hidden_before_mul = k @ w2
    hidden = F.silu(gate_before_act) * hidden_before_mul
    
    dhidden = v @ w1.transpose(-1, -2)
    dhidden_before_mul = dhidden * F.silu(gate_before_act)
    dgate = dhidden * hidden_before_mul
    dgate_before_act = pt_silu_backprop(dgate, gate_before_act)
    
    w1_grad = (hidden * lr1).transpose(-1, -2) @ v
    w0_grad = (k * lr0).transpose(-1, -2) @ dgate_before_act
    w2_grad = (k * lr2).transpose(-1, -2) @ dhidden_before_mul
    
    if norm_grad:
        w0_grad /= N
        w1_grad /= N
        w2_grad /= N
    
    # Update with Newton-Schulz
    w0 = w0 + pt_zeropower(w0_grad, muon_steps)
    w1 = w1 + pt_zeropower(w1_grad, muon_steps)
    w2 = w2 + pt_zeropower(w2_grad, muon_steps)
    
    # Weight normalization
    w0 = w0 / (w0.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
    w1 = w1 / (w1.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
    w2 = w2 / (w2.norm(dim=1, keepdim=True) + 1e-5) * w2_norm
    
    # Apply to query
    output = F.silu(q @ w0) * (q @ w2) @ w1
    return output

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load actual weights
    weights_path = "/home/kaiser/projects/vgg-ttt/vggttt_jax/vggttt.msgpack"
    with open(weights_path, "rb") as f:
        variables = serialization.msgpack_restore(f.read())
    jax_params = variables["params"]
    gb0 = jax_params["aggregator"]["global_blocks"]["layers_0"]["attn"]
    
    # w0/w1/w2 shape: [num_heads, d_in, d_h]
    w0_np = np.array(gb0["w0"])  # [16, 64, 256]
    w1_np = np.array(gb0["w1"])  # [16, 256, 64]
    w2_np = np.array(gb0["w2"])  # [16, 64, 256]
    print(f"w0: {w0_np.shape}, w1: {w1_np.shape}, w2: {w2_np.shape}")
    
    BH = 16  # 1 batch * 16 heads
    N = 64   # sequence length
    d = 64   # head_dim
    dh = 256 # hidden_dim (mlp_ratio=4 * head_dim=64)
    
    # Random q, k, v, lr
    q_np = np.random.randn(BH, N, d).astype(np.float32)
    k_np = np.random.randn(BH, N, d).astype(np.float32)
    v_np = np.random.randn(BH, N, d).astype(np.float32)
    lr_np = np.random.randn(BH, N, 1).astype(np.float32)
    lr0_np = lr_np
    lr1_np = lr_np * 0.9
    lr2_np = lr_np * 1.1
    
    # Expand w0/w1/w2 to [BH, d, dh] (B=1, so just the heads)
    w0_bh = w0_np  # already [16, 64, 256] = [num_heads, d, dh]
    w1_bh = w1_np  # [16, 256, 64]
    w2_bh = w2_np  # [16, 64, 256]
    
    # ---- PyTorch CPU TTT ----
    w0_pt = torch.from_numpy(w0_bh.copy())
    w1_pt = torch.from_numpy(w1_bh.copy())
    w2_pt = torch.from_numpy(w2_bh.copy())
    q_pt = torch.from_numpy(q_np)
    k_pt = torch.from_numpy(k_np)
    v_pt = torch.from_numpy(v_np)
    lr0_pt = torch.from_numpy(lr0_np)
    lr1_pt = torch.from_numpy(lr1_np)
    lr2_pt = torch.from_numpy(lr2_np)
    
    print("\n--- PyTorch CPU TTT ---")
    pt_out = pt_ttt_cpu(w0_pt, w1_pt, w2_pt, q_pt, k_pt, v_pt, lr0_pt, lr1_pt, lr2_pt, muon_steps=5, norm_grad=False)
    print(f"Output shape: {pt_out.shape}, max_abs: {pt_out.abs().max().item():.6e}")
    
    # ---- JAX TTT ----
    print("\n--- JAX TTT ---")
    q_jax = jnp.array(q_np)
    k_jax = jnp.array(k_np)
    v_jax = jnp.array(v_np)
    lr0_jax = jnp.array(lr0_np)
    lr1_jax = jnp.array(lr1_np)
    lr2_jax = jnp.array(lr2_np)
    w0_jax = jnp.array(w0_bh.copy())
    w1_jax = jnp.array(w1_bh.copy())
    w2_jax = jnp.array(w2_bh.copy())
    
    jax_out = ttt_apply(w0_jax, w1_jax, w2_jax, q_jax, k_jax, v_jax, lr0_jax, lr1_jax, lr2_jax, num_steps=1, muon_steps=5)
    print(f"Output shape: {jax_out.shape}, max_abs: {float(jnp.max(jnp.abs(jax_out))):.6e}")
    
    diff = np.max(np.abs(pt_out.numpy() - np.array(jax_out)))
    rel_diff = diff / (pt_out.abs().max().item() + 1e-8)
    print(f"\nMax Abs Diff: {diff:.6e}")
    print(f"Relative Diff: {rel_diff:.6e}")
    
    # ---- Debug Newton-Schulz ----
    print("\n--- Newton-Schulz Comparison ---")
    G_np = np.random.randn(BH, d, d).astype(np.float32)
    G_pt = torch.from_numpy(G_np.copy())
    G_jax = jnp.array(G_np.copy())
    
    ns_pt = pt_zeropower(G_pt, 5)
    ns_jax = zeropower_via_newtonschulz5(G_jax, 5)
    
    ns_diff = np.max(np.abs(ns_pt.numpy() - np.array(ns_jax)))
    print(f"Newton-Schulz Max Abs Diff: {ns_diff:.6e}")
    
    # ---- Debug gradient computation ----
    print("\n--- Gradient Computation Comparison ---")
    w0_g = torch.from_numpy(w0_bh.copy())
    w1_g = torch.from_numpy(w1_bh.copy())
    w2_g = torch.from_numpy(w2_bh.copy())
    
    # PyTorch grad
    gate_before_act = k_pt @ w0_g
    hidden_before_mul = k_pt @ w2_g
    hidden = F.silu(gate_before_act) * hidden_before_mul
    dhidden = v_pt @ w1_g.transpose(-1, -2)
    dhidden_before_mul = dhidden * F.silu(gate_before_act)
    dgate = dhidden * hidden_before_mul
    sigma = torch.sigmoid(gate_before_act)
    dgate_before_act = dgate * sigma * (1 + gate_before_act * (1 - sigma))
    
    pt_w1_grad = (hidden * lr1_pt).transpose(-1, -2) @ v_pt
    pt_w0_grad = (k_pt * lr0_pt).transpose(-1, -2) @ dgate_before_act
    pt_w2_grad = (k_pt * lr2_pt).transpose(-1, -2) @ dhidden_before_mul
    
    # JAX grad
    jax_w0_g, jax_w1_g, jax_w2_g = manual_grads(
        jnp.array(w0_bh.copy()), jnp.array(w1_bh.copy()), jnp.array(w2_bh.copy()),
        k_jax, v_jax, lr0_jax, lr1_jax, lr2_jax
    )
    
    diff_w0g = np.max(np.abs(pt_w0_grad.numpy() - np.array(jax_w0_g)))
    diff_w1g = np.max(np.abs(pt_w1_grad.numpy() - np.array(jax_w1_g)))
    diff_w2g = np.max(np.abs(pt_w2_grad.numpy() - np.array(jax_w2_g)))
    print(f"w0_grad Diff: {diff_w0g:.6e}")
    print(f"w1_grad Diff: {diff_w1g:.6e}")
    print(f"w2_grad Diff: {diff_w2g:.6e}")

if __name__ == "__main__":
    main()
