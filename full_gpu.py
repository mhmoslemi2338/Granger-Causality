from typing import Optional, Tuple, Dict
import torch

def schur_torch(A1):
    Q_total = torch.eye(A1.size()[0], dtype=A1.dtype, device=A1.device)
    
    T = A1.clone()

    for _ in range(30000):
        # 1. Perform QR decomposition on the current matrix
        # We use torch.linalg.qr, as torch.qr is deprecated.
        Q, R = torch.linalg.qr(T)
        
        # 2. Recombine as R @ Q
        # This new matrix is similar to the original A
        # and converges to the upper-triangular Schur form T.
        T = R @ Q
        
        # 3. Accumulate the orthogonal transformations
        Q_total = Q_total @ Q

    return T, Q_total



def _vecF(M: torch.Tensor) -> torch.Tensor:
    # Fortran-style vec (stack columns)
    return M.transpose(0, 1).contiguous().reshape(-1, 1)

def _unvecF(v: torch.Tensor, n: int, m: int) -> torch.Tensor:
    # Inverse of _vecF
    return v.reshape(m, n).transpose(0, 1).contiguous()

def lyapslv_torch(A: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    Solve discrete-time Lyapunov via real Schur:
        X = A X A^T + Q
    using the column-wise block back-substitution (as in your NumPy/Scipy code).

    Args:
        A: (n,n) real tensor
        Q: (n,n) real tensor
        schur_fn: callable returning (T, U) where A = U @ T @ U.T (real Schur)

    Returns:
        X: (n,n) real tensor
    """
    device = A.device
    dtype  = A.dtype
    n = A.shape[0]

    # Real Schur
    T, U = schur_torch(A)  # both (n,n)

    # Flip signs to make diag(U) >= 0 where possible
    s = torch.sign(torch.real(torch.diag(U)))
    s = torch.where(s == 0, torch.ones_like(s), s)
    S = torch.diag(s)

    U = U @ S
    T = S.T @ T @ S  # S is ±1 diagonal so S.T==S

    # Transform Q
    Qs = -(U.T @ Q @ U)

    Xs = torch.zeros((n, n), dtype=dtype, device=device)

    j = n
    # small tolerance to detect 2x2 blocks in quasi-upper-triangular T
    tol = torch.finfo(dtype).eps * 10

    while j > 0:
        j1 = j
        # detect 2x2 block if subdiagonal entry is (numerically) nonzero
        if j == 1:
            bsiz = 1
        elif torch.abs(T[j-1, j-2]) > tol:
            bsiz = 2
            j = j - 1  # include the (j-1)-th column as part of this 2x2 block
        else:
            bsiz = 1

        bsizn = bsiz * n

        # Kronecker system: (kron(T_jj, T) - I) vec(X(:, j..j1)) = rhs
        Tjj = T[j-1:j1, j-1:j1]                               # (bsiz, bsiz)
        Ajj = torch.kron(Tjj, T) - torch.eye(bsizn, dtype=dtype, device=device)

        rhs = _vecF(Qs[:, j-1:j1])                            # (bsizn, 1)

        # Add coupling to already-computed columns (to the right)
        if j1 < n:
            add_term = T @ (Xs[:, j1:n] @ T[j-1:j1, j1:n].T)  # (n, bsiz)
            rhs = rhs + _vecF(add_term)

        v = -torch.linalg.solve(Ajj, rhs)                     # (bsizn, 1)

        # Fill current column(s)
        Xs[:, j-1] = v[:n, 0]
        if bsiz == 2:
            Xs[:, j1-1] = v[n:bsizn, 0]

        j -= 1

    # Transform back
    X = U @ Xs @ U.T
    return X





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






def var_to_autocov_torch(
    A: torch.Tensor,
    SIG: torch.Tensor,
    error: float = 1e-3,
    q: Optional[int] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Torch port of your NumPy snippet.
    Returns (info, A1, SIG1) so you can keep going with the rest of your pipeline.
      - A:   (n, n, p)
      - SIG: (n, n)
    """

    device = A.device
    dtype  = A.dtype

    acdectol = error

    n, n1, p = A.shape
    assert n == n1, "A must be (n, n, p)"
    nn1, nn2 = SIG.shape
    assert (nn1, nn2) == (n, n), "SIG must be (n, n)"

    pn1 = (p - 1) * n
    pn  = p * n

    # --- info dict (torch tensors for NaNs on the right device) ---
    info = {
        "rho":       torch.tensor(float("nan"), device=device),
        "iters":     torch.tensor(float("nan"), device=device),
        "acrelerr":  torch.tensor(float("nan"), device=device),
        "acminlags": torch.tensor(float("nan"), device=device),
        "aclags":    torch.tensor(float("nan"), device=device),
    }

    G = None  # placeholder, as in your snippet

    # Build companion-like A1
    # A_top: (n, p*n) by concatenating A[:,:,k] across columns
    A_top = torch.cat([A[:, :, k] for k in range(p)], dim=1)  # (n, p*n)

    if pn1 > 0:
        # A_bottom: ((p-1)*n, p*n) = [I_(pn1), 0]
        A_bottom = torch.cat(
            [
                torch.eye(pn1, dtype=dtype, device=device),
                torch.zeros((pn1, n), dtype=dtype, device=device),
            ],
            dim=1,
        )
        A1 = torch.cat([A_top, A_bottom], dim=0)  # (p*n, p*n)
    else:
        # p == 1: companion reduces to A_top (n x n)
        A1 = A_top

    # Spectral radius: max |eig(A1)|
    eigvals, _ = torch.linalg.eig(A1)  # complex dtype if needed
    info["rho"] = eigvals.abs().max().real  # real scalar tensor

    # SIG1 block matrix:
    # [[SIG,            0_(n x pn1)]
    #  [0_(pn1 x n),    0_(pn1 x pn1)]]
    SIG1 = torch.zeros((n + pn1, n + pn1), dtype=SIG.dtype, device=device)
    SIG1[:n, :n] = SIG

    G1 = lyapslv_torch(A1, -SIG1)

    
    acminlags = torch.ceil(torch.log(torch.tensor(acdectol, dtype=dtype, device=device))
                           / torch.log(torch.tensor(float(info['rho']), dtype=dtype, device=device)))
    info['acminlags'] = int(acminlags.item())
    info['aclags'] = info['acminlags']

    if q is None:
        q = info['aclags']



    # Ensure plain ints
    q = int(q)
    q1 = q + 1

    # ---- Build G_part with Fortran-order semantics: reshape(G1[:n,:], (n,n,p), order='F')
    # In MATLAB/NumPy(F), this corresponds to taking consecutive n-column blocks.
    # G_part[:, :, k] = G1[:n, k*n:(k+1)*n]
    blocks = [G1[:n, k*n:(k+1)*n] for k in range(p)]
    G_part = torch.stack(blocks, dim=2)                         # (n, n, p)

    # Pad to length q1 along the 3rd dim
    if q1 - p > 0:
        G_tail = torch.zeros((n, n, q1 - p), dtype=dtype, device=device)
        G = torch.cat((G_part, G_tail), dim=2)                  # (n, n, q1)
    else:
        G = G_part[:, :, :q1]

    # B = vstack( zeros(( (q1-p)*n, n )), G1[:, -n:] )
    top_zeros = torch.zeros(((q1 - p) * n, n), dtype=dtype, device=device) if (q1 - p) > 0 \
                else torch.zeros((0, n), dtype=dtype, device=device)
    B = torch.vstack((top_zeros, G1[:, -n:]))                   # ((q1)*n, n)

    # A2d = reshape(A, (n, pn), order='F') i.e., concatenate A[:,:,k] along columns
    A2d = torch.cat([A[:, :, k] for k in range(p)], dim=1)      # (n, pn)

    # Main loop
    for k in range(p, q + 1):
        r = q1 - k
        # G[:, :, k] = A2d @ B[r*n : r*n + pn, :]
        G[:, :, k] = A2d @ B[r * n : r * n + pn, :]
        # B[(r-1)*n : r*n, :] = G[:, :, k]
        B[(r - 1) * n : r * n, :] = G[:, :, k]

    return G




def autocov_to_var_torch(G: torch.Tensor) -> torch.Tensor:
    """
    PyTorch version of autocov_to_var.

    Args:
        G (torch.Tensor): shape (n, n, q1)
    Returns:
        SIG (torch.Tensor): residual covariance (n, n)
    """

    n, _, q1 = G.shape
    q = q1 - 1
    qn = q * n

    G0 = G[:, :, 0]

    GF = G[:, :, 1:].permute(0, 2, 1).reshape(n, qn).T
    G_flipped = torch.flip(G[:, :, 1:], dims=[2])
    GB = G_flipped.permute(1, 2, 0).contiguous().view(n, qn).t()



    AF = torch.zeros((n, qn), dtype=G.dtype, device=G.device)
    AB = torch.zeros((n, qn), dtype=G.dtype, device=G.device)

    k = 1
    r = q - k
    kf = torch.arange(1, k * n+1, device=G.device)
    kb = torch.arange(r * n+1, qn+1, device=G.device)

    G0_inv = torch.linalg.inv(G0)

    AF[:, kf-1] = GB[kb-1, :] @ G0_inv
    AB[:, kb-1] = GF[kf-1, :] @ G0_inv

    for k in range(2, q + 1):


        res = GB[(r - 1) * n : r * n, :] - AF[:, kf-1] @ GB[kb-1, :]
        res2 = G0 - AB[:, kb-1] @ GB[kb-1, :]
        AAF = res @ torch.linalg.inv(res2)

        res = GF[(k - 1) * n : k * n, :] - AB[:, kb-1] @ GF[kf-1, :]
        res2 = G0 - AF[:, kf-1] @ GF[kf-1, :]
        AAB = res @ torch.linalg.inv(res2)

        AFPREV = AF[:, kf-1]
        ABPREV = AB[:, kb-1]

        r = q - k
        kf = torch.arange(1, k * n+1, device=G.device)
        kb = torch.arange(r * n+1, qn+1, device=G.device)

        AF[:, kf-1] = torch.cat([AFPREV - AAF @ ABPREV, AAF], dim=1)
        AB[:, kb-1] = torch.cat([AAB, ABPREV - AAB @ AFPREV], dim=1)

    SIG = G0 - AF @ GF
    return SIG




def autocov_to_pwcgc_torch(G: torch.Tensor, SIG: torch.Tensor) -> torch.Tensor:
    """
    PyTorch version of autocov_to_pwcgc.
    
    Args:
        G (torch.Tensor): Autocovariance sequence of shape (n, n, q)
        SIG (torch.Tensor): Residual covariance matrix of shape (n, n)
    
    Returns:
        torch.Tensor: Pairwise conditional Granger causality matrix (n, n)
    """
    device = G.device
    dtype = G.dtype
    n = G.shape[0]

    F = torch.full((n, n), float('nan'), dtype=dtype, device=device)
    LSIG = torch.log(torch.diag(SIG))

    for j in range(n):
        # Indices excluding j
        jo = torch.cat((torch.arange(0, j, device=device), torch.arange(j + 1, n, device=device)))

        # Sub-autocovariance sequence excluding variable j
        G_sub = G[jo][:, jo, :]

        # Compute submodel covariance
        SIGj = (autocov_to_var_torch((G_sub)))
        # SIGj = torch.tensor(autocov_to_var(np.array(G_sub)))
        LSIGj = torch.log(torch.diag(SIGj))

        # Fill F
        for ii in range(len(jo)):
            i = jo[ii]
            F[i, j] = LSIGj[ii] - LSIG[i]

    return F




def calc_F(X,p):
    p=3
    A, SIG, E = tsdata_to_var_torch(X, p)
    G= var_to_autocov_torch(A,SIG, q = 100)
    F = autocov_to_pwcgc_torch(G,SIG)
    F.fill_diagonal_(0)
    return F

