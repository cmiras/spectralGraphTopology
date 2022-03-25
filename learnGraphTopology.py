from utilscc import*
import numpy as np
from ObjectiveFunction import*
from time import*
def learn_k_component_graph (S, is_data_matrix = FALSE, k = 1, w0 = "naive", lb = 0, ub = 1e4, alpha = 0,\
                                    beta = 1e4, beta_max = 1e6, fix_beta = TRUE, rho = 1e-2, m = 7,\
                                    maxiter = 1e4, abstol = 1e-6, reltol = 1e-4, eigtol = 1e-9,\
                                    record_objective = False, record_weights = False, verbose = True):
  if (is_data_matrix or S.shape[0]!=S.shape[1]) {
    A = build_initial_graph(S, m = m)
    D = np.diag(.5 * (np.sum(A,axis=1)+ np.sum(A,axis=0))
    L = D - .5 * (A + A.T)
    S = np.linalg.pinv(L)
    is_data_matrix = True
  }
  # number of nodes
  n = S.shape[0]
  # l1-norm penalty factor
  H = alpha * (2 * diag(n) - matrix(1, n, n))
  K = S + H
  # find an appropriate inital guess
  if (is_data_matrix)
    Sinv = L
  else
    Sinv = np.linalg.pinv(S)
  # if w0 is either "naive" or "qp", compute it, else return w0
  w0 = w_init(w0, Sinv)
  # compute quantities on the initial guess
  Lw0 = La(w0)
  U0 = laplacian_U_update(Lw = Lw0, k = k)
  lambda0 = laplacian_lambda_update(lb = lb, ub = ub, beta = beta, U = U0,\
                                     Lw = Lw0, k = k)
  beta_seq = [beta]
  time_seq = [0]
  start_time = time()
  for i in np.arange(maxiter)
    w = laplacian_w_update(w = w0, Lw = Lw0, U = U0, beta = beta,\
                            lambd = lambda0, K = K)
    Lw = La(w)
    U = laplacian_U_update(Lw = Lw, k = k)
    lambd = laplacian_lambda_update(lb = lb, ub = ub, beta = beta, U = U,\
                                      Lw = Lw, k = k)
    # check for convergence
    werr = abs(w0 - w)
    has_w_converged = min(werr <= .5 * reltol * (w + w0)) or min(werr <= abstol)
    time_seq.append(time()-start_time)
    if not(fix_beta):
      eigvals=np.linalg.eigh(Lw)[0]
      n_zero_eigenvalues = np.sum(abs(eigvals) < eigtol)
      if (k <= n_zero_eigenvalues):
        beta = (1 + rho) * beta
      elif (k > n_zero_eigenvalues):
        beta = beta / (1 + rho)
      if (beta > beta_max):
        beta = beta_max
      beta_seq.append(beta)
    }
    if has_w_converged:
      break
    # update estimates
    w0 = w
    U0 = U
    lambda0 = lambd
    Lw0 = Lw
  # compute the adjancency matrix
  Aw = Ad(w)
  results = {Laplacian : Lw, Adjacency : Aw, w : w, "lambd" : lambd, "U" : U,\
                 "elapsed_time" : time_seq, "convergence" : has_w_converged,\
                  "beta_seq" : beta_seq}
  return results

def learn_cospectral_graph(S, lambd, k = 1, is_data_matrix = FALSE, w0 = "naive", alpha = 0,\
                                   beta = 1e4, beta_max = 1e6, fix_beta = TRUE, rho = 1e-2, m = 7,\
                                   maxiter = 1e4, abstol = 1e-6, reltol = 1e-4, eigtol = 1e-9,\
                                   record_objective = False, record_weights = False, verbose = True):
  if (is_data_matrix or S.shape[0] != S.shape[1]):
    A = build_initial_graph(S, m = m)
    D = np.diag(.5 * (np.sum(A,axis=1) + np.sum(A,axis=0)))
    L = D - .5 * (A + A.T)
    S = np.linalg.pinv(L)
    is_data_matrix = True
  # number of nodes
  n = S.shape[0]
  # l1-norm penalty factor
  H = alpha * (2 * np.eye(n)- np.ones(n, n))
  K = S + H
  # find an appropriate inital guess
  if (is_data_matrix)
    Sinv = L
  else
    Sinv = np.linalg.pinv(S)
  # if w0 is either "naive" or "qp", compute it, else return w0
  w0 = w_init(w0, Sinv)
  # compute quantities on the initial guess
  Lw0 = La(w0)
  U0 = laplacian_U_update(Lw = Lw0, k = k)
  beta_seq = [beta]
  time_seq = [0]
  start_time = time()
  for i in np.arange(maxiter):
    w = laplacian_w_update(w = w0, Lw = Lw0, U = U0, beta = beta,\
                            lambd = lambd, K = K)
    Lw = La(w)
    U = laplacian_U_update(Lw = Lw, k = k)
    # check for convergence
    werr = abs(w0 - w)
    has_w_converged = min(werr <= .5 * reltol * (w + w0)) or min(werr <= abstol)
    time_seq.append(time() - start_time)
    if not(fix_beta):
      eigvals = np.linalg.eigh(Lw)[0]
      n_zero_eigenvalues = np.sum(abs(eigvals) < eigtol)
      if (k <= n_zero_eigenvalues):
        beta = (1 + rho) * beta
      elif (k > n_zero_eigenvalues):
        beta = beta / (1 + rho)
      if (beta > beta_max):
        beta = beta_max
      beta_seq.append(beta)
    }
    if (has_w_converged)
      break
    # update estimates
    w0 = w
    U0 = U
    Lw0 = Lw
  }
  # compute the adjancency matrix
  Aw = Ad(w)
  results = {"Laplacian" : Lw, "Adjacency" : Aw, "w" : w, "lambd" : lambd, "U" : U,\
                  "elapsed_time" : time_seq, "convergence" : has_w_converged,\
                  "beta_seq" : beta_seq}
  return results
