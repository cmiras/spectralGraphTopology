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
