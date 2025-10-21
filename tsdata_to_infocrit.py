    

import numpy as np

def infocrit(L, k, m):
    if m - k - 1 <= 0:
        aic = np.nan
    else:
        aic = -2 * L + 2 * k * (m / (m - k - 1))  # corrected AIC
        # If you want uncorrected AIC, use: aic = -2 * L + 2 * k

    bic = -2 * L + k * np.log(m)
    return aic, bic


def _demean(X):
    """Subtract mean along a given axis (keeps dims)."""
    n, m,N = X.shape
    Y = X.swapaxes(1, 2).reshape(n, -1)
    row_means = np.mean(Y, axis=1, keepdims=True)
    Y = Y - row_means
    row_stds = np.std(Y, axis=1, keepdims=True, ddof=1)
    Y = np.divide(Y, row_stds, out=np.zeros_like(Y), where=row_stds!=0)
    return Y[:,:,None]




def tsdata_to_infocrit(X, morder, verb=True):
    """
    Python port of MVGC-style info-criteria via LWR recursion.

    Parameters
    ----------
    X : ndarray, shape (n, m, N)
        n variables x m observations per trial x N trials.
    morder : int
        Maximum VAR order to evaluate (positive scalar).
    verb : bool
        If True, prints progress per order.

    Returns
    -------
    aic : ndarray, shape (morder,)
        AIC for orders 1..morder (NaN where estimation failed).
    bic : ndarray, shape (morder,)
        BIC for orders 1..morder (NaN where estimation failed).
    moaic : int
        Order minimizing AIC (1-based). NaN if all AIC are NaN.
    mobic : int
        Order minimizing BIC (1-based). NaN if all BIC are NaN.
    """
    X = X[:,:,None] 


    X = np.asarray(X, dtype=float)
    assert X.ndim == 3, "X must have shape (n, m, N)"
    n, m, N = X.shape

    # No constant term; normalise by demeaning along time within each trial
    X = _demean(X)

    # Model order handling (scalar only, matching your MATLAB branch)
    q = int(morder)
    assert q > 0, "model order must be a positive scalar"
    q1 = q + 1

    # # Build lagged data stack: XX has lag index 0..q along axis=1
    q = np.max(morder)
    q1 = q + 1
    XX = np.zeros((n, q1, m + q, N))
    for k in range(q + 1):
        XX[:, k, k:k+m, :] = X


    aic = np.full(q, np.nan, dtype=float)
    bic = np.full(q, np.nan, dtype=float)

    q1n = q1*n
    I = np.eye(n)


    # Initial whitening
    E0 = X.reshape(n, m, order='F')
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
    AF = np.zeros((n, q1n), dtype=float)
    AB = np.zeros((n, q1n), dtype=float)



    k = 1
    kn = k * n
    M = N * (m - k)

    kk = np.arange(1, k + 1)
    kf = np.arange(1,kn+1)
    kb = np.arange(q1n-kn+1,q1n+1)

    AF = np.zeros((n, q1n))
    AF[:, kf-1] = IC  # Assigns IC to the first 'kn' columns

    AB = np.zeros((n, q1n))
    AB[:, kb-1] = IC  # Assigns IC to the last 'kn' columns





    while k<=q:
    # for i in range(2):


        result = np.reshape(XX[:, kk-1, k:m, :], (kn, M),order='F')
        EF = AF[:,kf-1] @ result

        result = np.reshape(XX[:, kk-1, k-1:m-1, :], (kn, M),order='F')
        EB = AB[:,kb-1] @ result



        


        # Cholesky of EF*EF' (lower triangular)
        L_F = np.linalg.cholesky(EF @ EF.T)
        L_B = np.linalg.cholesky(EB @ EB.T)

        # Equivalent of (L_F \ EF) * (L_B \ EB)'
        R = np.linalg.solve(L_F, EF) @ np.linalg.solve(L_B, EB).T





        kp = k
        k+=1
        kn = k*n
        M  = N*(m-k)

                




        kk = np.arange(1, k + 1)
        kf = np.arange(1, kn + 1)
        kb = np.arange(q1n-kn+1,q1n+1)

        AFPREV = AF[:, kf - 1]
        ABPREV = AB[:, kb - 1]


        # Cholesky decompositions
        LF = np.linalg.cholesky(I - R @ R.T)
        LB = np.linalg.cholesky(I - R.T @ R)


        L1 = np.linalg.cholesky(I - R @ R.T)
        AF[:, kf-1] = np.linalg.solve(L1, AFPREV - R @ ABPREV)


        L2 = np.linalg.cholesky(I - R.T @ R)
        AB[:, kb-1] = np.linalg.solve(L2, ABPREV - R.T @ AFPREV)



        E = np.linalg.solve(AF[:, :n], EF)


        DSIG = np.linalg.det((E @ E.T) / (M - 1))

        aic_tmp, bic_tmp = infocrit(-(M/2)*np.log(DSIG),(kp)*n*n,M)


        aic[kp-1],bic[kp-1] = aic_tmp, bic_tmp 



    # Choose optimal orders (ignore NaNs)
    morder = np.arange(1, morder + 1)
    idx_aic = np.nanargmin(aic)
    moaic = morder[idx_aic]

    idx_bic = np.nanargmin(bic)
    mobic = morder[idx_bic]


    return aic, bic, moaic, mobic



