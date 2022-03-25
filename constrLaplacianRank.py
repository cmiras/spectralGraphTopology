from utilscc import*
from time import*
import numpy as np
from qpsolvers import *
def build_initial_graph(Y,m):
    #if wellunderstood create the m nearest neighboor directed graph
    n=Y.shape[0]
    A=np.zeros(0,n,n)
    E=pairwise_matrix_rownorm(Y)
    for i in np.arange(0,n):
        sorted_index=np.sort(E[i])
        j_sweep = sorted_index[1,m+1]
        den = m*E[i][sorted_index[m+2]]-np.sum(E[i][j_sweep])
        ei= E[i,sorted_index[m+1]]
        for j in j_sweep:
            A[i,j]=(ei-E[i,j])/den
    return A

def cluster_k_component_graph(Y, k = 1, m = 5, lmd = 1, eigtol = 1e-9,
                                      edgetol = 1e-6, maxiter = 1000):
  time_seq =[0]
  start_time = time()
  A = build_initial_graph(Y, m)
  n = A.shape[1]
  S = np.ones(n,n)/n
  DS = np.diag(.5 * (np.sum(S,axis=1) + np.sum(S,axis=0))
  LS =  DS - .5 * (S + S.T)
  DA =- np.diag(.5 * (np.sum(A,axis=1) + np.sum(A,axis=0))
  LA <- DA - .5 * (A + A.T)
  if (k == 1)
    F = np.linalg.eigh(LA)[1][:,0:k]
  else
    F = np.linalg.eigh(LA)[1][:,0:k]
  # bounds for variables in the QP solver
  bvec = [1, np.array(np.arange(n+1)]
  Amat = [np.arange(1,n+1),np.eyes(n)]
  lmd_seq = np.array([lmd])
  #pb <- progress::progress_bar$new(format = "<:bar> :current/:total  eta: :eta  lambda: :lmd  null_eigvals: :null_eigvals",
    #                               total = maxiter, clear = FALSE, width = 100)
  for ii in np.arange(1,maxiter+1):
    V = pairwise_matrix_rownorm(F)
    for i in np.arange(n):
      p = A[i,: ] - .5 * lmd * V[i,:]
      qp = solve_qp(P, q, G, h, A, b)#quadprog::solve.QP(Dmat = diag(n), dvec = p, Amat = Amat, bvec = bvec, meq = 1)
      S[i, ] = qp#qp$solution
      DS = np.diag(.5 * (np.sum(S,axis=1) + np.sum(S,axis=0))
      LS =  DS - .5 * (S + S.T)
    F = np.linalg.eigh(LS)[1][:, 0:k]
    eig_vals = np.linalg.eigh(LS)[0]
    n_zero_eigenvalues = sum(abs(eig_vals) < eigtol)
    time_seq.append(time() - start_time)
    #pb$tick(token = list(lmd = lmd, null_eigvals = n_zero_eigenvalues))progress_bar replace with tqdm if time
    if k < n_zero_eigenvalues:
      lmd = .5 * lmd
    elif (k > n_zero_eigenvalues):
      lmd = 2 * lmd
    else:
      break
    lmd_seq.append(lmd)
  }
  LS[abs(LS) < edgetol] <- 0
  AS <- np.diag(np.diagonal(LS)) - LS
  return {"Laplacian" : LS, "Adjacency" : AS, "eigenvalues" : eig_vals,"lmd_seq" : lmd_seq, "elapsed_time" : time_seq)}
}
