import numpy as np

# Computes the Adjacency linear operator which maps a vector of weights into
# a valid Adjacency matrix.
# @param w weight vector of the graph
# @return Aw the Adjacency matrix
def Ad(v):
    """take an p(p-1)//2 array and return the adjacency matrice"""
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

# Computes the Laplacian linear operator which maps a vector of weights into a valid Laplacian matrix.
# @param w weight vector of the graph
# @return Lw the Laplacian matrix
def La(v):
    a = -Ad(v)
    for k in range(a.shape[0]):
        a[k][k]=-np.sum(a[k])
    return a

def Lstar(M):
  """
  Compute the adjoint operator of L
  """
  N = M.shape[1]
  k = (N * (N - 1)) // 2
  j, l = 0, 1
  w = np.zeros(k)
  for i in np.arange(k):
    w[i] = M[j, j] + M[l, l] - (M[l, j] + M[j, l])
    if (l == (N - 1)):
        j += 1
        l = j + 1
    else:
      l += 1
  return w

#Computes the matrix form of the composition of the operators Lstar and
# L, i.e., Lstar o L.
#
# @param n number of columns/rows
# @return M the composition of Lstar and L
def Mmat(n):
  e = np.zeros(n)
  M = np.zeros(n, n)
  e[0] = 1
  M[0] = Lstar(La(e))
  for j in np.arange(1,n):
    e[j - 1] = 0
    e[j] = 1
    M[j] = Lstar(L(e))
  return M.T

def Astar(M):
  N = M.shape[1]
  k = (N * (N - 1))//2
  j = 0
  l = 1
  w=np.zeros(k)

  for i in np.arange(k):
    w[i] = M[l, j] + M[j, l]
    if l == (N - 1):
      j+=1
      l = j+1
    else:
      l+=1
  return w


# Computes the matrix form of the composition of the operators Astar and
# A, i.e., Astar o A.
def Pmat(n):
  e = np.zeros(n)
  M = np.zeros(n, n)
  e[0] = 1;
  M[0] = Astar(Ad(e))
  for j in np.arange(1,n):
    e[j - 1] = 0
    e[j] = 1
    M[j] = Astar(A(e))
  return M.T

def vec(M):
  return M.T.flatten()

def vecLmat(n):
  ncols = (n * (n - 1))//2
  nrows = n * n

  e = np.zeros(ncols)
  R = np.zeros(nrows,ncols)
  e[0] = 1;
  R[0] = vec(L(e));
  for j in np.arange(1,ncols):
    e[j - 1] = 0;
    e[j] = 1;
    R[j] = vec(L(e));
  return R.T




#Computes the inverse of the L operator.

# @param M Laplacian matrix
# @return w the weight vector of the graph
def Linv(M):
  n = M.shape[0]
  return np.concatenate([-M[i][i+1:] for i in np.arange(n)])

#get the n(n-1)//2 vector from the laplacian(or A?)
#M is laplacian
#w is weight vector
def Ainv(M):
  N = M.shape[0]
  return np.concatenate([M[i][i+1:] for i in np.arange(n)])
