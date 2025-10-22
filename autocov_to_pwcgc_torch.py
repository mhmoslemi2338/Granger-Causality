import torch
from autocov_to_var_torch import autocov_to_var_torch

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
