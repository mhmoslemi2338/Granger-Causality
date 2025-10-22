
import numpy as np
from tsdata_to_infocrit import _demean



def tsdata_to_var(X,p):

    # X  = np.array(X).T # X must be numpy array in shape of (num var, num rows)
    X  = np.array(X) # X must be numpy array in shape of (num var, num rows)


    X = X[:,:,None] 
    X = np.asarray(X, dtype=float)
    n, m, N = X.shape

    M = N*(m-p)

    p1 = p+1
    pn = p*N
    p1n = p1*n



    A   = np.nan
    SIG = np.nan
    E   = np.nan

    X = _demean(X, normalize=False)



    XX = np.zeros((n, p1, m + p, N))
    for k in range(p + 1):
        XX[:, k, k:k+m, :] = X

    I = np.eye(n)



    # Initial whitening
    E0 = X.reshape(n, N*m, order='F')
    EE = E0 @ E0.T

    try:
        L0 = np.linalg.cholesky(EE)
        IC = np.linalg.inv(L0)  # inverse covariance square root

    except np.linalg.LinAlgError:
        # Tiny jitter fallback if covariance not PD
        jitter = 1e-10 * np.trace(EE) / n
        L0 = np.linalg.cholesky(EE + jitter * np.eye(n))
        IC = np.linalg.inv(L0)

    # Allocate AR coefficient blocks
    AF = np.zeros((n, p1n), dtype=float)
    AB = np.zeros((n, p1n), dtype=float)



    k = 1
    kn = k * n
    M = N * (m - k)

    kk = np.arange(1, k + 1)
    kf = np.arange(1,kn+1)
    kb = np.arange(p1n-kn+1,p1n+1)

    AF = np.zeros((n, p1n))
    AF[:, kf-1] = IC  # Assigns IC to the first 'kn' columns

    AB = np.zeros((n, p1n))
    AB[:, kb-1] = IC  # Assigns IC to the last 'kn' columns



    while k<=p:
    # while k<=1:
    # for i in range(2):

        result = np.reshape(XX[:, kk-1, k:m, :], (kn, M),order='F')
        EF = AF[:,kf-1] @ result

        result = np.reshape(XX[:, kk-1, k-1:m-1, :], (kn, M),order='F')
        EB = AB[:,kb-1] @ result

        # return EF, EB

        # Cholesky of EF*EF' (lower triangular)
        L_F = np.linalg.cholesky(EF @ EF.T)
        L_B = np.linalg.cholesky(EB @ EB.T)

        # Equivalent of (L_F \ EF) * (L_B \ EB)'
        R = np.linalg.solve(L_F, EF) @ np.linalg.solve(L_B, EB).T


        k+=1
        kn = k*n
        M  = N*(m-k)
        kk = np.arange(1, k + 1)
        kf = np.arange(1, kn + 1)
        kb = np.arange(p1n-kn+1,p1n+1)

        AFPREV = AF[:, kf - 1]
        ABPREV = AB[:, kb - 1]


        # # Cholesky decompositions
        # LF = np.linalg.cholesky(I - R @ R.T)
        # LB = np.linalg.cholesky(I - R.T @ R)


        L1 = np.linalg.cholesky(I - R @ R.T)
        AF[:, kf-1] = np.linalg.solve(L1, AFPREV - R @ ABPREV)



        L2 = np.linalg.cholesky(I - R.T @ R)
        AB[:, kb-1] = np.linalg.solve(L2, ABPREV - R.T @ AFPREV)


    A0 = AF[:,0:n]

    A = np.linalg.solve(A0, AF[:, n:p1n]) 
    A = -A.reshape((n, n, p), order='F')

    E = np.linalg.solve(A0, EF)


    SIG = (E @ E.T) / (M - 1)   # unbiased covariance

    E = E.reshape((n, m - p, N), order='F')

    return A, SIG, np.squeeze(E)