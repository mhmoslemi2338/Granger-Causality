import numpy as np


def autocov_to_var_optimized(G):

    n,_,q1 = G.shape
    q = q1-1
    qn = q*n

    G0 = G[:,:,0]

    GF = np.reshape(G[:, :, 1:], (n, qn), order='F').T
    GB = np.reshape(
        np.transpose(np.flip(G[:, :, 1:], axis=2), (0, 2, 1)),
        (qn, n),
        order='F'
    )


    # Initialize forward and backward coefficient matrices
    AF = np.zeros((n, qn))
    AB = np.zeros((n, qn))

    # Initialize recursion
    k = 1  # model order

    r = q - k
    kf = np.arange(1, k * n+1)       # forward indices (MATLAB 1:k*n -> Python 0:k*n)
    kb = np.arange(r * n+1, qn+1)      # backward indices (MATLAB r*n+1:qn -> Python r*n:qn)

    # # Assign coefficients (matrix right division in MATLAB -> multiplication by inv(G0) in Python)
    AF[:, kf-1] = GB[kb-1, :] @ np.linalg.inv(G0)
    AB[:, kb-1] = GF[kf-1, :] @ np.linalg.inv(G0)



    for k in range(2,q+1):
        
        res = (GB[(r - 1)*n : r*n, :] - AF[:, kf-1] @ GB[kb-1, :])
        res2 = (G0 - AB[:, kb-1] @ GB[kb-1, :])
        # AAF = res @ np.linalg.inv(res2)
        AAF = np.linalg.solve(res2.T, res.T).T

        res = GF[(k-1)*n:k*n,:]- AB[:,kb-1] @ GF[kf-1,:]
        res2 = G0-AF[:,kf-1] @ GF[kf-1,:]
        AAB = res @ np.linalg.inv(res2)


        AFPREV = AF[:,kf-1]
        ABPREV = AB[:,kb-1]

        r = q-k
        kf = np.arange(1, k * n+1)
        kb = np.arange(r*n+1,qn+1)

        # AF[:,kf-1] = np.hstack((AFPREV - AAF @ ABPREV, AAF))
        AF[:, 0:(k-1)*n] = AFPREV - AAF @ ABPREV
        AF[:, (k-1)*n:k*n] = AAF


        # AB[:,kb-1] = np.hstack((AAB, ABPREV - AAB @ AFPREV))
        AB[:, r*n : r*n + n] = AAB
        AB[:, r*n + n : qn] = ABPREV - AAB @ AFPREV


    SIG = G0-AF@GF
    return SIG





