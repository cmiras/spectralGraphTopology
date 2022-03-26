import numpy as np

def blockDiagCpp(matrices):
    """matrcies:square np array list
    return big np array diag by block with blocks given in matrices"""
    n=len(matrices)
    sizes=[k.shape[0] for k in matrices]
    N=sum(sizes)
    if min(k.shape[0]==k.shape[1] for k in matrices):
        raise ValueError('all matrices are not square.')
    blockDiag=np.zeros(N,N)
    i=0
    for m in matrices:
        n=m.shape[0]
        for a in range(n):
            for b in range(n):
                blockDiag[i+a][i+b]=m[a][b]
        i+=m.shape[0]
    return blockDiag

def pairwise_matrix_rownorm(M):
    """
    Compute the matrix E where Eij is ||x_i - x_j||**2
    """
    n = M.shape[0]
    V = np.zeros([n, n])
    for i in range(n):
        for j in range(i+1, n):
            V[i][j]=np.linalg.norm(M[i]-M[j])**2
    V+=V.T
    return V

def metrics(A,B,eps):
    p=A.shape[0]
    tp=0
    fp=0
    fn=0
    tn=0
    for i in range(p):
        for j in range(p):
            if abs(A[i][j])>eps:
                if abs(B[i][j])>eps:
                    tp+=1
                else:
                    fn+=1
            elif abs(B[i][j])>eps:
                fp+=1
            else:
                tn+=1
    fscore = 2 * tp / (2 * tp + fn + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn);
    return [fscore, recall, specificity, accuracy]

def relative_error(A, B):
    """
    Compute the relative error between the matrices A and B, taking B as reference
    """
    return np.linalg.norm(A - B) / np.linalg.norm(B)


def upper_view_vec(M):
    t = 0
    p = M.shape[1]
    v = np.concatenate([M[i][i+1:p] for i in np.arange(p-1)])
    return v

def fscore(A,B,eps=1e-4):
    return metrics(A, B, eps)[0]

def prial(Ltrue, Lest, Lscm):
    """
    Compute the prial metric between Ltrue, Lest and Lscm.
    More details #TODO
    """
    return 100 * (1 - (np.linalg.norm(Lest - Ltrue) / np.linalg.norm(Lscm - Ltrue)**2))
