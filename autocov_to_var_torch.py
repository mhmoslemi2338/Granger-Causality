import torch



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

