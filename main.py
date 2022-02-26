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
