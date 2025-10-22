
from scipy.linalg import schur
import numpy as np


def lyapslv(A, Q):


    n = A.shape[0]

    T, U = schur(A, output='real')



    # Flip signs to make diag(U) nonnegative (where possible)
    s = np.sign(np.real(np.diag(U)))
    s[s == 0] = 1  # leave zeros as +1 to avoid NaNs
    S = np.diag(s)

    # Apply similarity inside the Schur decomposition:
    # A = U T U^T = (U S) (S^T T S) (U S)^T
    U = U @ S
    T = S.T @ T @ S  # since S is diagonal ±1, S.T == S




    Qs = -U.T @ Q @ U

    Xs = np.zeros((n, n), dtype=T.dtype)

    j = n




    while j > 0:
        j1 = j


        if j == 1:
            bsiz = 1
        elif T[j-1,j-2] != 0:
            bsiz = 2
            j = j-1
        else:
            bsiz = 1
        bsizn = bsiz*n


        # Kronecker system: (kron(T_jj, T) - I) vec(X(:, j..j1)) = rhs
        Tjj = T[j-1:j1, j-1:j1]          # (bsiz x bsiz)
        Ajj = np.kron(Tjj, T) - np.eye(bsizn, dtype=T.dtype)


        rhs = Qs[:, j-1:j1].reshape((bsizn, 1), order='F')

        # Add coupling to already-computed columns (if any to the right)
        if j1 < n:
            # T * ( Xs(:, j1+1:n) * T(j..j1, j1+1:n)' )
            add_term = T @ (Xs[:, j1:n] @ T[j-1:j1, j1:n].T)
            rhs = rhs + add_term.reshape((bsizn, 1), order='F')



        v = -np.linalg.solve(Ajj, rhs)
        Xs[:,j-1] = np.squeeze(v[:n])


        if bsiz == 2:
            Xs[:, j1-1] = np.squeeze(v[n:bsizn])

        j-=1




    X = U @ Xs @ U.T


    return X
