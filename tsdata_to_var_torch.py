

import torch

def _torch_demean(X: torch.Tensor, normalize: bool = False):
    """
    X: (n, m, N)
    Demean across the 'sample' axis collectively over trials, i.e., per variable (row)
    over all m*N samples. Matches MATLAB-style:
       Y = X(:,:); Y = Y - mean(Y,2)*ones(1,N*m); reshape back.
    """
    n, m, N = X.shape
    Y = X.reshape(n, m * N)
    mu = Y.mean(dim=1, keepdim=True)            # (n,1)
    Y = Y - mu
    if normalize:
        # unbiased=False here (population std) to mirror common MATLAB behavior unless you want ddof=1
        sd = Y.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-12)
        Y = Y / sd
    return Y.reshape(n, m, N)


def _chol_with_jitter(S, max_tries=5, jitter_scale=1e-10):
    """
    Robust Cholesky with trace-based jitter fallback.
    """
    n = S.shape[-1]
    trace_S = torch.trace(S).clamp_min(0.0)
    jitter = 0.0
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(S + jitter * torch.eye(n, dtype=S.dtype, device=S.device))
        except RuntimeError:
            # Increase jitter (geometric)
            jitter = jitter_scale * (trace_S / max(n, 1)) * (10.0 if jitter > 0 else 1.0)
            if jitter == 0.0:  # handle trace 0
                jitter = jitter_scale
    # Final attempt may still fail and raise, which is fine (surface the error)
    return torch.linalg.cholesky(S + jitter * torch.eye(n, dtype=S.dtype, device=S.device))


def tsdata_to_var_torch(X, p: int, dtype=torch.float64, device=None):

    """
    PyTorch implementation of your tsdata_to_var.

    Args:
        X: array-like (n, m) or (n, m, N). If (n, m), a singleton trial dim is added.
        p: VAR order
        dtype: torch dtype (default float64 for stability)
        device: torch device (inferred from X if torch.Tensor; else cpu)

    Returns:
        A: (n, n, p)  VAR coefficients (Torch tensor)
        SIG: (n, n)   residual covariance (Torch tensor, unbiased)
        E: (n, m-p, N) residuals (Torch tensor)
    """
    # ---- ingest & shape ----
    if isinstance(X, torch.Tensor):
        if device is None:
            device = X.device
        X = X.to(dtype=dtype, device=device)
    else:
        device = torch.device("cpu") if device is None else device
        X = torch.as_tensor(X, dtype=dtype, device=device)

    if X.ndim == 2:
        X = X.unsqueeze(-1)  # (n, m, 1)
    elif X.ndim != 3:
        raise ValueError("X must be (n, m) or (n, m, N)")

    n, m, N = X.shape
    if p < 1 or p >= m:
        raise ValueError("p must be >=1 and < m")

    # ---- demean (no normalize) ----
    X = _torch_demean(X, normalize=False)

    # Prebuild lagged design tensor XX: (n, p+1, m+p, N), with XX[:,k,k:k+m,:] = X
    p1 = p + 1
    XX = torch.zeros((n, p1, m + p, N), dtype=dtype, device=device)
    for k in range(p1):
        XX[:, k, k:k+m, :] = X

    I = torch.eye(n, dtype=dtype, device=device)

    # ---- Initial whitening using all samples across trials ----
    # E0 is vec of X across trials/time in MATLAB 'F' sense. We can use C-order with consistent math:
    E0 = X.reshape(n, N * m)  # (n, Nm)
    EE = E0 @ E0.T            # (n, n)

    L0 = _chol_with_jitter(EE)
    IC = torch.linalg.inv(L0)

    # ---- Allocate AR coefficient blocks ----
    p1n = p1 * n
    AF = torch.zeros((n, p1n), dtype=dtype, device=device)
    AB = torch.zeros((n, p1n), dtype=dtype, device=device)

    # k = 1 init
    k = 1
    kn = k * n
    M = N * (m - k)  # current number of stacked samples per (EF, EB)

    # index helpers (1-based in original; we’ll use 0-based with same shapes)
    # kf: first kn cols; kb: last kn cols of the p1n block
    kk = torch.arange(1, k+1, device=device)
    kf = torch.arange(1, kn+1, device=device)
    kb = torch.arange(p1n - kn+1, p1n+1, device=device)

    # Place IC:
    AF[:, kf-1] = IC
    AB[:, kb-1] = IC


    while k <= p:
    # while k <= 1:

        block_F = XX[:, kk-1, k:m, :]                      # (n, k, m-k, N)
        result_F = block_F.permute(1, 0, 2, 3).contiguous().view(kn, M)
        EF = AF[:, kf-1] @ result_F                       # (n, M)

        block_B = XX[:, kk-1, k-1:m-1, :]                  # (n, k, m-k, N)
        result_B = block_B.permute(1, 0, 2, 3).contiguous().view(kn, M)
        EB = AB[:, kb-1] @ result_B                      # (n, M)


        # return EF, EB
        # R = (L_F \ EF) * (L_B \ EB)'
        L_F = _chol_with_jitter(EF @ EF.T)
        L_B = _chol_with_jitter(EB @ EB.T)
        R = torch.linalg.solve(L_F, EF) @ torch.linalg.solve(L_B, EB).T  # (n,n)

        # Update to next k
        k += 1
        kn = k * n
        M = N * (m - k)
        kk = torch.arange(1, k+1, device=device)
        kf = torch.arange(1, kn+1, device=device)
        kb = torch.arange(p1n-kn+1,p1n+1, device=device)



        # kf_next = torch.arange(0, kn_next, device=device)
        # kb_next = torch.arange(p1n - kn_next, p1n, device=device)

        # Keep previous active blocks
        AFPREV = AF[:, kf-1]  # first kn columns
        ABPREV = AB[:, kb-1] # last kn columns

        # AF(:,1:kn_next) = inv(chol(I - R R^T)) * (AFPREV - R * ABPREV)
        # AB(:,end-kn_next+1:end) = inv(chol(I - R^T R)) * (ABPREV - R^T * AFPREV)
        LF = _chol_with_jitter(I - R @ R.T)
        LB = _chol_with_jitter(I - R.T @ R)

        AF[:, kf-1] = torch.linalg.solve(LF, AFPREV - R @ ABPREV)
        AB[:, kb-1] = torch.linalg.solve(LB, ABPREV - R.T @ AFPREV)

    # Extract A0 and A
    A0 = AF[:, :n]                              # (n,n)
    # Solve A0 * X = AF[:, n:]  -> X = A0^{-1} AF[:, n:]
    A_flat = torch.linalg.solve(A0, AF[:, n:])  # (n, p*n)

    # Reshape to (n,n,p) with lag-major stacking (equivalent to MATLAB/NumPy order='F')
    A_lags = []
    for lag in range(p):
        A_lags.append(A_flat[:, lag * n : (lag + 1) * n])  # (n,n)
    A = -torch.stack(A_lags, dim=2)  # (n,n,p)

    # Residuals from the last EF step (whitened AF applied to data at order p)
    # But we need residuals in the original (not whitened by A0): E = A0^{-1} * EF_last

    E = torch.linalg.solve(A0, EF)  # (n, last_M)

    # Unbiased covariance SIG = (E E^T) / (M - 1)  with M corresponding to last_EF sample count
    denom = max(M - 1, 1)
    SIG = (E @ E.T) / denom

    # Reshape E back to (n, m-p, N).
    # We stacked columns as (time, trial) in C-order; cols=M = (m-p)*N
    # Arrange to (n, m-p, N) with time-major second axis:
    E = E.view(n, (m - p), N)

    return A, (SIG), (E)
