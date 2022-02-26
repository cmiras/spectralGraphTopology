import numpy as np
def A(v):
    """take an p(p-1)//2 array and return the adjacency matrices"""
    p=1
    while (p*(p-1))//2!=v.shape[0]:
        p+=1
    a = np.zeros([p,p])
    s=0
    for nb in range(p-1,0,-1):
        i=p-1-nb
        for j in range(i+1,p):
            a[i][j] = v[s+j-i-1]
            a[j][i] = v[s+j-i-1]
        s += nb
    return a

def L(v):
    a = -A(v)
    for k in range(a.shape[0]):
        a[k][k]=-np.sum(a[k])
    return a
def fscore(A,B):
    p=A.shape[0]
    tp=0
    fp=0
    fn=0
    for i in range(p):
        for j in range(p):
            if A[i][j]:
                if B[i][j]:
                    tp+=1
                else:
                    fn+=1
            elif B[i][j]:
                fp+=1
    return (2*tp)/(2*tp+fn+fp)
print(L(np.array([2,3,2,2,2,2])))
def pairwise_matrix_rownorm(M):
    n=M.shape(0)
    V=np.zero([n,n])
    for i in range(n):
        for j in range(i+1,n):
            V[i][j]=np.linalg.norm(M[i]-M[j])**2
    V+=V.T
    return V
def blockDiagCpp(matrices):
    """matrcies:square np array list
    return big np array diag by block with blocks given in matrices"""
    n=matrices.shape[0]
    sizes=[k.shape[0] for k in matrices]
    N=sum(k)
    if min(k.shape[0]==k.shape[1] for k in matrices):
        raise ValueError('all matrices are not square.')
    blockDiag=np.zeros([N,N])
    i=0
    for m in matrices:
        n=m.shape[0]
        for a in range(n):
            for b in range(n):
                blockDiag[i+a][i+b]=m[a][b]
        i+=m.shape[0]
    return blockDiag
        
def build_initial_graph(Y,m):
    n=Y.shape[0]
    A=np.zeros(n,n)
def learn_k_component_graph(S,is_data_matrix=False, k=1,w0="naive",lb=0,ub=10**4,alpha=0,):
    return None

def w_init(w0, Sinv):
    """
    Params: 
        w0: straing ("qp" or "naive")
        Sinv: numpy array
    """
    if type(w0) == str:
        if w0 == "qp":
            R = vecLmat(Sinv.shape)
            #qp = cvx.solvers.qp(np.outer(R, R), )
            w0 = 0
        elif w0 == "naive":
            w0 = Linv(Sinv)
            w0[w0 < 0] = 0
    return w0