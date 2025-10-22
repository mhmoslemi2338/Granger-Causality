import torch
from typing import Optional, Tuple, Dict
from lyapslv_torch import lyapslv_torch



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
