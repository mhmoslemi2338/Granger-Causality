
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



@torch.no_grad()
def _hessenberg_torch(A: torch.Tensor):
    """Unblocked Householder Hessenberg reduction.
    Returns H, Q with A = Q @ H @ Q.T, H upper-Hessenberg, Q orthogonal/unitary.
    Works on CPU or GPU depending on A.device. A must be real square.
    """
    assert A.ndim == 2 and A.shape[0] == A.shape[1]
    n = A.shape[0]
    H = A.clone()
    Q = torch.eye(n, dtype=A.dtype, device=A.device)

    for k in range(n - 2):
        x = H[k+1:, k]
        normx = torch.linalg.norm(x)
        if normx == 0:
            continue
        # Householder vector
        sign = torch.sign(x[0]) if x[0] != 0 else x.new_tensor(1.0)
        v = x.clone()
        v[0] = v[0] + sign * normx
        v = v / torch.linalg.norm(v)

        # Apply from left: H[k+1:, k:] -= 2 v (v^T H[k+1:, k:])
        H[k+1:, k:] -= 2.0 * (v[:, None] @ (v[None, :] @ H[k+1:, k:]))

        # Apply from right: H[:, k+1:] -= 2 (H[:, k+1:] v) v^T
        H[:, k+1:] -= 2.0 * ((H[:, k+1:] @ v[:, None]) @ v[None, :])

        # Accumulate Q: Q[:, k+1:] -= 2 (Q[:, k+1:] v) v^T
        Q[:, k+1:] -= 2.0 * ((Q[:, k+1:] @ v[:, None]) @ v[None, :])

    return H, Q

@torch.no_grad()
def _wilkinson_shift_2x2(B: torch.Tensor):
    """Wilkinson shift for the trailing 2x2 block B."""
    # B is 2x2 real
    tr = B[0,0] + B[1,1]
    det = B[0,0]*B[1,1] - B[0,1]*B[1,0]
    # eigenvalues of B
    disc = tr*tr - 4*det
    if disc >= 0:
        s = torch.sqrt(disc)
        mu1 = 0.5*(tr + s)
        mu2 = 0.5*(tr - s)
        # choose the eigenvalue closer to B[1,1]
        return mu1 if torch.abs(mu1 - B[1,1]) < torch.abs(mu2 - B[1,1]) else mu2
    else:
        # complex pair: pick the real part of the pair (Rayleigh-like)
        return 0.5*tr

@torch.no_grad()
def schur_torch_(A: torch.Tensor, tol=1e-8, max_iters=10000):
    """
    Real Schur-like decomposition via:
      1) Hessenberg reduction
      2) Shifted QR with deflation (single/Wilkinson shift)
    Returns T (quasi upper-triangular) and Q with A ≈ Q @ T @ Q.T

    Works on CPU or GPU. For best GPU speed, batch many matrices and
    run this loop over the batch or write a batched variant.
    """
    assert A.ndim == 2 and A.shape[0] == A.shape[1]
    n = A.shape[0]
    if n == 1:
        return A.clone(), torch.ones_like(A)

    # 1) Hessenberg reduction
    H, Qh = _hessenberg_torch(A)
    T = H
    Q = Qh

    I = torch.eye(n, dtype=A.dtype, device=A.device)

    m = n  # active size
    it = 0
    # Helper to zero tiny subdiagonals and deflate
    def deflate(T, m, tol):
        while m > 1:
            # test subdiagonal at m-1
            tiny = torch.abs(T[m-1, m-2]) <= tol*(torch.abs(T[m-2, m-2]) + torch.abs(T[m-1, m-1]))
            if tiny:
                T[m-1, m-2] = T.new_tensor(0.0)
                m -= 1
            else:
                break
        return m

    m = deflate(T, m, tol)

    while m > 1 and it < max_iters:
        # Choose shift (2x2 Wilkinson when possible; else Rayleigh)
        if m >= 2:
            B = T[m-2:m, m-2:m]
            mu = _wilkinson_shift_2x2(B)
        else:
            mu = T[m-1, m-1]

        # Shifted QR step on leading m×m block
        # QR of (T - mu I)
        Qk, Rk = torch.linalg.qr(T[:m, :m] - mu * I[:m, :m])
        T[:m, :m] = Rk @ Qk + mu * I[:m, :m]
        Q[:, :m] = Q[:, :m] @ Qk

        it += 1
        m = deflate(T, m, tol)

    return T, Q









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





