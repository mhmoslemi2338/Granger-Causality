
import numpy as np
from lyapslv import lyapslv

def var_to_autocov(A,SIG):


    acdectol  = 1e-8

    n, n1, p = A.shape

    pn1 = (p - 1) * n
    nn1, nn2 = SIG.shape

    # --- info struct (as a dict) ---
    info = {
        "rho":       np.nan,
        "iters":     np.nan,
        "acrelerr":  np.nan,
        "acminlags": np.nan,
        "aclags":    np.nan,
    }

    G = None  


    A_top = np.concatenate([A[:, :, k] for k in range(p)], axis=1)  # (n, p*n)
    if pn1 > 0:
        A_bottom = np.hstack([np.eye(pn1), np.zeros((pn1, n))])     # ((p-1)n, pn)
        A1 = np.vstack([A_top, A_bottom])                           # (pn, pn) where pn = p*n
    else:
        # p == 1 case: companion reduces to just A_top (n x n)
        A1 = A_top

    eigvals = np.linalg.eigvals(A1)
    info["rho"] = np.max(np.abs(eigvals))


    SIG1 = np.block([
        [SIG,                   np.zeros((n,   pn1), dtype=SIG.dtype)],
        [np.zeros((pn1, n),     dtype=SIG.dtype),     np.zeros((pn1, pn1), dtype=SIG.dtype)]
    ])


    G1 = lyapslv(A1, -SIG1)

    # res = A1 @ G1 @ A1.T - G1 + SIG1
    # info["acrelerr"] = np.linalg.norm(res, ord=2) / np.linalg.norm(SIG1,ord=2)
    info['acminlags'] = np.ceil(np.log(acdectol)/np.log(info['rho']))
    info['aclags'] = info['acminlags']

    q = info['aclags']
    q1 = q+1

    n,_,p = A.shape
    pn = p*n


    G_part = np.reshape(G1[:n, :], (n, n, p), order='F')
    G = np.concatenate((G_part, np.zeros((n, n, int(q1 - p)))), axis=2)
    B = np.vstack((np.zeros((int(q1 - p) * n, n)), G1[:, -n:]))
    A = np.reshape(A, (n, pn), order='F')

    q = int(q)
    q1 = int(q1)
    for k in range(p,q+1):
        r = q1-k
        G[:,:,k] = A @ B[r*n:r*n+pn,:]
        B[(r-1)*n:r*n,:] = G[:,:,k]


    return G