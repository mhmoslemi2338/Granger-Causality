import torch
from typing import Optional, Tuple

# ---------- small helpers ----------

def _vecF(M: torch.Tensor) -> torch.Tensor:
    return M.transpose(0, 1).contiguous().reshape(-1, 1)

def _unvecF(v: torch.Tensor, n: int, m: int) -> torch.Tensor:
    return v.reshape(m, n).transpose(0, 1).contiguous()

def _torch_demean(X: torch.Tensor, normalize: bool = False):
    n, m, N = X.shape
    Y = X.reshape(n, m * N)
    mu = Y.mean(dim=1, keepdim=True)
    Y = Y - mu
    if normalize:
        sd = Y.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-12)
        Y = Y / sd
    return Y.reshape(n, m, N)

def _chol_with_jitter(S, max_tries=5, jitter_scale=1e-10):
    n = S.shape[-1]
    trace_S = torch.trace(S).clamp_min(0.0)
    jitter = 0.0
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(S + jitter * torch.eye(n, dtype=S.dtype, device=S.device))
        except RuntimeError:
            jitter = jitter_scale * (trace_S / max(n, 1)) * (10.0 if jitter > 0 else 1.0)
            if jitter == 0.0:
                jitter = jitter_scale
    return torch.linalg.cholesky(S + jitter * torch.eye(n, dtype=S.dtype, device=S.device))

def _project_stable(A: torch.Tensor, margin: float = 0.99):
    """
    Soft stability projection to enforce ||A||_2 <= margin.
    Keeps things well-posed during training; differentiable.
    """
    # largest singular value
    smax = torch.linalg.svdvals(A)[0]
    scale = torch.clamp(margin / smax, max=1.0)
    return A * scale

# ---------- differentiable Stein (discrete Lyapunov) via Smith/doubling ----------

@torch.no_grad()
def _smith_iters_needed(tol: float) -> int:
    # safe default upper bound on iterations if user only passes tol
    # doubling shrinks 'residual' roughly quadratically in exponent; 64 is already generous
    return 64

def smith_dlyap(A: torch.Tensor,
                Q: torch.Tensor,
                max_iters: Optional[int] = None,
                tol: float = 1e-10,
                project: bool = True) -> torch.Tensor:
    """
    Solve:  X = A X A^T + Q
    Smith/doubling iteration: X_{k+1} = X_k + H_k;  H_{k+1} = G_k H_k G_k^T;  G_{k+1} = G_k G_k
    All ops are matmuls/adds -> fully differentiable.
    """
    assert A.shape[0] == A.shape[1] == Q.shape[0] == Q.shape[1]
    A_use = _project_stable(A) if project else A

    X = torch.zeros_like(Q)
    G = A_use
    H = Q.clone()

    if max_iters is None:
        max_iters = _smith_iters_needed(tol)

    for _ in range(max_iters):
        X_next = X + H
        # early stop (keeps graph small but is still differentiable wrt completed steps)
        if torch.linalg.norm(H) <= tol * (1.0 + torch.linalg.norm(X_next)):
            X = X_next
            break
        X = X_next
        H = G @ H @ G.transpose(-1, -2)
        G = G @ G
    return X

def stein_solve(A: torch.Tensor, Q: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Solve the Stein equation:  X - A X A^T = Q   (standard in VAR autocov)
    by rewriting as:           X = A X A^T + (-Q)
    """
    return smith_dlyap(A, -Q, **kwargs)

# ---------- your tsdata_to_var, kept differentiable ----------

def tsdata_to_var_torch(X, p: int, dtype=torch.float64, device=None):
    if isinstance(X, torch.Tensor):
        device = X.device if device is None else device
        X = X.to(dtype=dtype, device=device)
    else:
        device = torch.device("cpu") if device is None else device
        X = torch.as_tensor(X, dtype=dtype, device=device)

    if X.ndim == 2:
        X = X.unsqueeze(-1)
    elif X.ndim != 3:
        raise ValueError("X must be (n,m) or (n,m,N)")

    n, m, N = X.shape
    if p < 1 or p >= m:
        raise ValueError("p must be >=1 and < m")

    X = _torch_demean(X, normalize=False)

    p1 = p + 1
    XX = torch.zeros((n, p1, m + p, N), dtype=dtype, device=device)
    for k in range(p1):
        XX[:, k, k:k+m, :] = X

    # initial whitening
    E0 = X.reshape(n, N*m)
    EE = E0 @ E0.T
    L0 = _chol_with_jitter(EE)
    IC = torch.linalg.inv(L0)

    I = torch.eye(n, dtype=dtype, device=device)
    p1n = p1 * n
    AF = torch.zeros((n, p1n), dtype=dtype, device=device)
    AB = torch.zeros((n, p1n), dtype=dtype, device=device)

    k = 1
    kn = k * n
    M = N * (m - k)
    kk = torch.arange(1, k+1, device=device)
    kf = torch.arange(1, kn+1, device=device)
    kb = torch.arange(p1n - kn + 1, p1n + 1, device=device)

    AF[:, kf-1] = IC
    AB[:, kb-1] = IC

    while k <= p:
        block_F = XX[:, kk-1, k:m, :]
        result_F = block_F.permute(1, 0, 2, 3).contiguous().view(kn, M)
        EF = AF[:, kf-1] @ result_F

        block_B = XX[:, kk-1, k-1:m-1, :]
        result_B = block_B.permute(1, 0, 2, 3).contiguous().view(kn, M)
        EB = AB[:, kb-1] @ result_B

        L_F = _chol_with_jitter(EF @ EF.T)
        L_B = _chol_with_jitter(EB @ EB.T)
        R = torch.linalg.solve(L_F, EF) @ torch.linalg.solve(L_B, EB).T

        k += 1
        kn = k * n
        M = N * (m - k)
        kk = torch.arange(1, k+1, device=device)
        kf = torch.arange(1, kn+1, device=device)
        kb = torch.arange(p1n - kn + 1, p1n + 1, device=device)

        AFPREV = AF[:, kf-1] if k > 1 else AF[:, :0]  # safe for k=1
        ABPREV = AB[:, kb-1] if k > 1 else AB[:, :0]

        LF = _chol_with_jitter(I - R @ R.T)
        LB = _chol_with_jitter(I - R.T @ R)
        AF[:, kf-1] = torch.linalg.solve(LF, AFPREV - R @ ABPREV)
        AB[:, kb-1] = torch.linalg.solve(LB, ABPREV - R.T @ AFPREV)

    A0 = AF[:, :n]
    A_flat = torch.linalg.solve(A0, AF[:, n:])  # (n, p*n)
    A_lags = [A_flat[:, lag*n:(lag+1)*n] for lag in range(p)]
    A = -torch.stack(A_lags, dim=2)  # (n,n,p)

    E = torch.linalg.solve(A0, EF)
    denom = max(M - 1, 1)
    SIG = (E @ E.T) / denom
    E = E.view(n, (m - p), N)
    return A, SIG, E

# ---------- VAR -> autocov with differentiable Stein solve ----------

def var_to_autocov_torch(
    A: torch.Tensor,
    SIG: torch.Tensor,
    error: float = 1e-3,
    q: Optional[int] = None,
) -> torch.Tensor:

    device, dtype = A.device, A.dtype
    n, _, p = A.shape
    pn1 = (p - 1) * n
    pn  = p * n

    # Companion-like A1
    A_top = torch.cat([A[:, :, k] for k in range(p)], dim=1)
    if pn1 > 0:
        A_bottom = torch.cat(
            [torch.eye(pn1, dtype=dtype, device=device),
             torch.zeros((pn1, n), dtype=dtype, device=device)],
            dim=1,
        )
        A1 = torch.cat([A_top, A_bottom], dim=0)  # (pn, pn)
    else:
        A1 = A_top  # (n, n) if p==1

    # Stein form we want: A1 G1 A1^T - G1 + SIG1 = 0  ==>  G1 = A1 G1 A1^T + (-SIG1)
    SIG1 = torch.zeros((A1.shape[0], A1.shape[0]), dtype=dtype, device=device)
    SIG1[:n, :n] = SIG

    G1 = smith_dlyap(A1, SIG1, max_iters=1000, tol=1e-8, project=False)
    # lyapslv_torch(A1,-SIG1)

    # choose q (lags)
    # acminlags ~= ceil(log(error) / log(rho))  — we avoid eig; use op-norm proxy
    # fall back to a safe number if user doesn't pass q
    if q is None:
        q = int(max(1, torch.ceil(torch.tensor(10.0, device=device)).item()))
    q1 = q + 1

    # G forward part: reshape like Fortran blocks
    blocks = [G1[:n, k*n:(k+1)*n] for k in range(p)]
    G_part = torch.stack(blocks, dim=2) if p > 0 else torch.zeros((n, n, 0), dtype=dtype, device=device)

    if q1 - p > 0:
        G = torch.cat((G_part, torch.zeros((n, n, q1 - p), dtype=dtype, device=device)), dim=2)
    else:
        G = G_part[:, :, :q1]

    # Build B and roll forward to fill remaining lags
    top_zeros = torch.zeros(((q1 - p) * n, n), dtype=dtype, device=device) if (q1 - p) > 0 else torch.zeros((0, n), dtype=dtype, device=device)
    B = torch.vstack((top_zeros, G1[:, -n:]))

    A2d = torch.cat([A[:, :, k] for k in range(p)], dim=1) if p > 0 else torch.zeros((n, 0), dtype=dtype, device=device)

    for k in range(p, q + 1):
        r = q1 - k
        if p > 0:
            G[:, :, k] = A2d @ B[r * n: r * n + pn, :]
        else:
            G[:, :, k] = torch.zeros((n, n), dtype=dtype, device=device)
        B[(r - 1) * n: r * n, :] = G[:, :, k]

    return G  # (n,n,q+1)

# ---------- autocov -> VAR (pairwise GC needs SIGj); keep differentiable ----------

def autocov_to_var_torch(G: torch.Tensor) -> torch.Tensor:
    n, _, q1 = G.shape
    q = q1 - 1
    qn = q * n

    G0 = G[:, :, 0]

    # Forward sequence (Fortran semantics)
    GF = G[:, :, 1:].permute(0, 2, 1).reshape(n, qn).T
    # Backward sequence (Fortran semantics)
    GB = torch.flip(G[:, :, 1:], dims=[2]).permute(1, 2, 0).contiguous().view(n, qn).t()

    AF = torch.zeros((n, qn), dtype=G.dtype, device=G.device)
    AB = torch.zeros((n, qn), dtype=G.dtype, device=G.device)

    k = 1
    r = q - k
    kf = torch.arange(1, k * n + 1, device=G.device)
    kb = torch.arange(r * n + 1, qn + 1, device=G.device)

    # Use solve instead of explicit inverse
    AF[:, kf-1] = torch.linalg.solve(G0.T, GB[kb-1, :].T).T
    AB[:, kb-1] = torch.linalg.solve(G0.T, GF[kf-1, :].T).T

    for k in range(2, q + 1):
        r = q - k + 1  # previous r in your code
        # AAF
        res  = GB[(r - 1) * n: r * n, :] - AF[:, kf-1] @ GB[kb-1, :]
        res2 = G0 - AB[:, kb-1] @ GB[kb-1, :]
        AAF  = torch.linalg.solve(res2.T, res.T).T  # res @ inv(res2)

        # AAB
        res  = GF[(k - 1) * n: k * n, :] - AB[:, kb-1] @ GF[kf-1, :]
        res2 = G0 - AF[:, kf-1] @ GF[kf-1, :]
        AAB  = torch.linalg.solve(res2.T, res.T).T

        AFPREV = AF[:, kf-1]
        ABPREV = AB[:, kb-1]

        r = q - k
        kf = torch.arange(1, k * n + 1, device=G.device)
        kb = torch.arange(r * n + 1, qn + 1, device=G.device)

        AF[:, kf-1] = torch.cat([AFPREV - AAF @ ABPREV, AAF], dim=1)
        AB[:, kb-1] = torch.cat([AAB, ABPREV - AAB @ AFPREV], dim=1)

    SIG = G0 - AF @ GF
    return SIG

# ---------- pairwise-conditional GC (differentiable) ----------

# def autocov_to_pwcgc_torch(G: torch.Tensor, SIG: torch.Tensor) -> torch.Tensor:
#     device, dtype = G.device, G.dtype
#     n = G.shape[0]
#     # F = torch.full((n, n), float('nan'), dtype=dtype, device=device)
#     F = torch.full((n, n), float(0), dtype=dtype, device=device)
#     LSIG = torch.log(torch.diag(SIG))
    

#     for j in range(n):
#         jo = torch.cat((torch.arange(0, j, device=device), torch.arange(j+1, n, device=device)))
#         G_sub = G[jo][:, jo, :]
#         SIGj  = autocov_to_var_torch(G_sub)
#         LSIGj = torch.log(torch.diag(SIGj))
#         F[jo, j] = LSIGj - LSIG[jo]
#     return F

def autocov_to_pwcgc_torch(G: torch.Tensor, SIG: torch.Tensor) -> torch.Tensor:
    device, dtype = G.device, G.dtype
    n = G.shape[0]
    F = torch.full((n, n), float('nan'), dtype=dtype, device=device)

    eps = torch.finfo(dtype).eps if dtype.is_floating_point else 1e-12
    LSIG = torch.log(torch.clamp(torch.diag(SIG), min=eps))  # <- safe log

    I = torch.eye(n, device=device, dtype=dtype)

    for j in range(n):
        jo = torch.cat((torch.arange(0, j, device=device),
                        torch.arange(j+1, n, device=device)))
        G_sub = G[jo][:, jo, :]

        SIGj = autocov_to_var_torch(G_sub)

        # minimal stabilization before log:
        SIGj = 0.5 * (SIGj + SIGj.T)  # symmetrize
        # add just-enough jitter if needed
        mineig = torch.linalg.eigvalsh(SIGj).min()
        if mineig <= 0:
            SIGj = SIGj + (-(mineig) + eps) * torch.eye(SIGj.shape[0], device=device, dtype=dtype)

        LSIGj = torch.log(torch.clamp(torch.diag(SIGj), min=eps))  # <- safe log

        F[jo, j] = LSIGj - LSIG[jo]

    return F

# ---------- glue: compute F end-to-end ----------

import torch




import torch

def reverse_yule_walker(A: torch.Tensor, Sigma: torch.Tensor, q: int = 50, tol: float = 1e-9, max_iters: int = 2000):
    """
    Reverse Yule–Walker: compute autocovariance sequence from VAR(p) coefficients and noise covariance.

    Args:
        A: (n, n, p)  VAR coefficient matrices
        Sigma: (n, n) innovation covariance
        q: number of lags to compute (default 50)
        tol: Lyapunov convergence tolerance
        max_iters: max iterations for Smith solver

    Returns:
        G: (n, n, q+1)  autocovariance sequence, where G[:,:,0] = covariance
    """
    n, _, p = A.shape
    device, dtype = A.device, A.dtype

    # --- Companion matrix ---
    Ap = torch.cat([A[:, :, k] for k in range(p)], dim=1)  # (n, n*p)
    if p == 1:
        A_comp = A[:, :, 0]
    else:
        top = torch.cat([Ap, torch.zeros((n, (p - 1) * n), device=device, dtype=dtype)], dim=1)
        bottom_left = torch.eye((p - 1) * n, device=device, dtype=dtype)
        bottom_right = torch.zeros(((p - 1) * n, n), device=device, dtype=dtype)
        bottom = torch.cat([bottom_left, bottom_right], dim=1)
        A_comp = torch.cat([top, bottom], dim=0)  # (p*n, p*n)

    # --- Lyapunov solve: G0 satisfies G0 = A G0 Aᵀ + Σ ---
    G0 = Sigma.clone()
    A11 = A_comp[:n, :n]
    for _ in range(max_iters):
        G_next = A11 @ G0 @ A11.T + Sigma
        if torch.norm(G_next - G0) < tol:
            break
        G0 = G_next
    G = torch.zeros((n, n, q + 1), device=device, dtype=dtype)
    G[:, :, 0] = G0

    # --- Recursively compute higher-order autocovariances ---
    for k in range(1, q + 1):
        acc = torch.zeros((n, n), device=device, dtype=dtype)
        for i in range(1, min(p, k) + 1):
            acc += A[:, :, i - 1] @ G[:, :, k - i]
        G[:, :, k] = acc

    return G


def autocov_torch(Y: torch.Tensor, q: int) -> torch.Tensor:
    """
    Compute autocovariance sequence Γ_k = Cov(Y_t, Y_{t-k})
    for k = 0, 1, ..., q.
    
    Args:
        Y: Tensor of shape (T, n), rows = time, cols = features.
        q: Number of lags to compute (non-negative int).
    
    Returns:
        G: Tensor of shape (n, n, q+1)
           where G[:, :, k] = Γ_k
    """
    T, n= Y.shape
    Y = Y - Y.mean(dim=0, keepdim=True)   # center data
    
    G = torch.zeros((n, n, q+1), dtype=Y.dtype, device=Y.device)
    
    for k in range(q + 1):
        Y_t = Y[k:]              # (T - k, n)
        Y_tk = Y[:T - k]         # (T - k, n)
        G[:, :, k] = (Y_t.t() @ Y_tk) / (T - k)
    
    return G

def pwcgc_differentiable(X, p = 3, q: Optional[int] = 20):
    """
    X: (n, m) or (n, m, N)
    Returns: F (n,n) differentiable wrt X (and through all steps)
    """
    A, SIG, _ = tsdata_to_var_torch(X, 20)
    
    G = autocov_torch(X.t(),q = 15)
    print(G.shape)
    F = autocov_to_pwcgc_torch(G, SIG)
    F.fill_diagonal_(0)
    return F


