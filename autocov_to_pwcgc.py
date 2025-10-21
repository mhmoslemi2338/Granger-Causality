import numpy as np
from autocov_to_var import autocov_to_var

def autocov_to_pwcgc(G,SIG):

    n = G.shape[0]
    F = np.ones((n,n)) * np.nan
    LSIG = np.log(np.diag(SIG))


    for j in range(1,n+1):

        jo = np.concatenate((np.arange(1, j), np.arange(j + 1, n + 1)))
        SIGj = autocov_to_var(G[np.ix_(jo-1, jo-1, np.arange(G.shape[2]))])
        LSIGj = np.log(np.diag(SIGj))
        for ii in range(1,n):
            i = jo[ii-1]
            F[i-1,j-1] = LSIGj[ii-1]-LSIG[i-1]
            

    return F