import numpy as np
# this module contains implementations of the algorithms
# proposed in L. Zhao et. al. "Optimization Algorithms for
# Graph Laplacian Estimation via ADMM and MM", IEEE Trans. Sign. Proc. 2019.

def get_incidence_from_adjacency(A):
    p = A.shape[0]
    m = np.sum(A>0)//2
    E=np.zeros(p,m)
    k=0
    for i in np.arange(p-1):
        for j in np.arange(i+1,p):
            if(A[i,j]>0):
                E[i][k]=1
                E[j][k]=-1
                k+=1
    return E

def learn_laplacian_gle_mm(S, A_mask = None, alpha = 0, maxiter = 10000, reltol = 1e-5,\
                                   record_objective = False, verbose = True):
    p=S.shape[0]
    Sinv=np.linalg.pinv(S)
    if A_mask==None:
        A_mask=np.ones(p,p)-np.diag(np.ones(p))
    #mask=>0    #what is Ainv
