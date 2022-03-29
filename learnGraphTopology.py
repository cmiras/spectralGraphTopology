from utilscc import *
import numpy as np
from ObjectiveFunction import *
from time import *
from operators import *
from BlockCoordinateDescent import *
from constrLaplacianRank import *
from GraphLaplacianEstimation import *

def learn_k_component_graph (S, is_data_matrix = False, k = 1, w0 = "naive", lb = 0, ub = 1e4, alpha = 0,\
                                    beta = 1e4, beta_max = 1e6, fix_beta = True, rho = 1e-2, m = 7,\
                                    maxiter = 1e4, abstol = 1e-6, reltol = 1e-4, eigtol = 1e-9,\
                                    record_objective = False, record_weights = False, verbose = True):
  """
  Learn the Laplacian and adjacency matrix corresponding to a k-component graph
  Params: #TODO
    S:
    is_data_matrix:
  Params:
    S: Either the original correlation matrix or the raw data matrix
    is_data_matrix: bool, if True then the correlations matrix is computed from S
    k: number of components of the final graph
    m: number of neighbors considered to build the matrix (only useful if is_data_matrix is true)
  """
  if (is_data_matrix or S.shape[0] != S.shape[1]):
    A = build_initial_graph(S, m = m)
    D = np.diag(.5 * (np.sum(A,axis=1) + np.sum(A,axis=0)))
    L = D - .5 * (A + A.T)
    S = np.linalg.pinv(L)
    is_data_matrix = True
  # number of nodes
  n = S.shape[0]
  # l1-norm penalty factor
  H = alpha * (2 *np.eye(n) - np.ones([n, n]))
  K = S + H
  # find an appropriate inital guess
  if (is_data_matrix):
    Sinv = L
  else:
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
  for i in np.arange(maxiter):
    #test_time = time()
    #test_total_time = time()
    w = laplacian_w_update(w = w0, Lw = Lw0, U = U0, beta = beta,\
                            lambd = lambda0, K = K, p=S.shape[0])
    #test_laplacian_w_update_time= time() - test_time
    #test_time = time()
    Lw = La(w)
    #test_La_time = time() - test_time
    #test_time = time()
    U = laplacian_U_update(Lw = Lw, k = k)
    #test_laplacian_U_update_time = time() - test_time
    #test_time = time()
    lambd = laplacian_lambda_update(lb = lb, ub = ub, beta = beta, U = U,\
                                      Lw = Lw, k = k)
    #test_laplacian_lambda_update_time = time() - test_time
    #test_time = time()
    # check for convergence
    werr = abs(w0 - w)
    has_w_converged = (np.all(werr <= .5 * reltol * (w + w0)) or np.all(werr <= abstol))
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
    if has_w_converged:
      break
    # update estimates
    w0 = w
    U0 = U
    lambda0 = lambd
    Lw0 = Lw
    """
    test_convergence_time = time() - test_time
    test_total_time = time() - test_total_time
    print('total time', test_total_time)
    print('total ratio (1):', (test_laplacian_w_update_time + test_La_time + test_laplacian_U_update_time + test_laplacian_lambda_update_time + test_convergence_time)/test_total_time)
    print('laplacian_w_update', test_laplacian_w_update_time/test_total_time*100)
    print('La', test_La_time/test_total_time*100)
    print('laplacian_U_update', test_laplacian_U_update_time/test_total_time*100)
    print('laplacian_lambda_update', test_laplacian_lambda_update_time/test_total_time*100)
    print('convergence', test_convergence_time/test_total_time*100)"""
  # compute the adjacency matrix
  Aw = Ad(w)
  results = {"Laplacian" : Lw, "Adjacency" : Aw, "w" : w, "lambd" : lambd, "U" : U,\
                 "elapsed_time" : time_seq, "convergence" : has_w_converged,\
                  "beta_seq" : beta_seq}
  return results

def learn_cospectral_graph(S, lambd, k = 1, is_data_matrix = False, w0 = "naive", alpha = 0,\
                                   beta = 1e4, beta_max = 1e6, fix_beta = True, rho = 1e-2, m = 7,\
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
  if (is_data_matrix):
    Sinv = L
  else:
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
    if (has_w_converged):
      break
    # update estimates
    w0 = w
    U0 = U
    Lw0 = Lw
  # compute the adjacency matrix
  Aw = Ad(w)
  results = {"Laplacian" : Lw, "Adjacency" : Aw, "w" : w, "lambd" : lambd, "U" : U,\
                  "elapsed_time" : time_seq, "convergence" : has_w_converged,\
                  "beta_seq" : beta_seq}
  return results

def learn_bipartite_graph(S, is_data_matrix = False, z = 0, nu = 1e4, alpha = 0.,
                                  w0 = "naive", m = 7, maxiter = 1e4, abstol = 1e-6, reltol = 1e-4,
                                  record_weights = False, verbose = True):
  if is_data_matrix or S.shape[0] != S.shape[1]:
    A = build_initial_graph(S, m = m)
    D =  np.diag(.5 * (np.sum(A,axis=1) + np.sum(A,axis=0)))
    L = D - .5 * (A + A.T)
    S = np.linalg.pinv(L)
    is_data_matrix = True
  # number of nodes
  n = S.shape[0]
  # note now that S is always some sort of similarity matrix
  J = np.ones([n,n])/n
  # l1-norm penalty factor
  H = alpha * (2 * np.eye(n) - np.ones([n, n]))
  K = S + H
  # compute initial guess
  if is_data_matrix:
    Sinv = L
  else:
    Sinv = np.linalg.pinv(S)
  # if w0 is either "naive" or "qp", compute it, else return w0
  w0 = w_init(w0, Sinv)
  Lips = 1 / np.linalg.eigh(La(w0) + J)[0][0]
  # compute quantities on the initial guess
  Aw0 = Ad(w0)
  V0 = bipartite_V_update(Aw0, z)
  psi0 = bipartite_psi_update(V0, Aw0)
  Lips_seq = [Lips]
  time_seq = [0]
  start_time = time()
  ll0 = bipartite_likelihood(Lw = La(w0), K = K, J = J)
  fun0 = ll0 + bipartite_prior(nu = nu, Aw = Aw0, psi = psi0, V = V0)
  fun_seq = [fun0]
  ll_seq = [ll0]
  for i in np.arange(maxiter):
    # we need to make sure that the Lipschitz constant is large enough
    # in order to avoid divergence
    while 1:
      # compute the update for w
      w = bipartite_w_update(w = w0, Aw = Aw0, V = V0, nu = nu, psi = psi0,
                              K = K, J = J, Lips = Lips)
      # compute the objective function at the updated value of w
      fun_t=bipartite_obj_fun(Aw = Ad(w), Lw = La(w), V = V0, psi = psi0,
                        K = K, J = J, nu = nu)
      """fun_t = tryCatch({#TODO
                   bipartite.obj_fun(Aw = A(w), Lw = L(w), V = V0, psi = psi0,
                                     K = K, J = J, nu = nu)
                 }, warning = function(warn) return(Inf), error = function(err) return(Inf)
               )"""
      # check if the previous value of the objective function is
      # smaller than the current one
      Lips_seq.append(Lips)
      if fun0 < fun_t:
        # in case it is in fact larger, then increase Lips and recompute w
        Lips = 2 * Lips
    else:
        # otherwise decrease Lips and get outta here!
        Lips = .5 * Lips
        if Lips < 1e-12:
          Lips = 1e-12
        break
    Lw = La(w)
    Aw = Ad(w)
    V = bipartite_V_update(Aw = Aw, z = z)
    psi = bipartite_psi_update(V = V, Aw = Aw)
    # compute negloglikelihood and objective function values
    ll = bipartite_likelihood(Lw = Lw, K = K, J = J)
    fun = ll + bipartite_prior(nu = nu, Aw = Aw, psi = psi, V = V)
    # save measurements of time and objective functions
    time_seq.append(time()- start_time)
    ll_seq.append(ll)
    fun_seq.append(fun)
    # compute the relative error and check the tolerance on the Adjacency
    # matrix and on the objective function
    # check for convergence
    werr = abs(w0 - w)
    has_w_converged = (np.all(werr <= .5 * reltol * (w + w0)) or np.all(werr <= abstol))
    if (has_w_converged):
      break
    # update estimates
    fun0 = fun
    w0 = w
    V0 = V
    psi0 = psi
    Aw0 = Aw
  results = {"Laplacian" : Lw, "Adjacency" : Aw, "obj_fun" : fun_seq, "loglike" : ll_seq, "w" : w,
                  "psi" : psi, "V" : V, "elapsed_time" : time_seq, "Lips" : Lips,
                  "Lips_seq" : Lips_seq, "convergence" : (i < maxiter), "nu" : nu}
  return results


def learn_bipartite_k_component_graph(S, is_data_matrix = False, z = 0, k = 1,\
                                              w0 = "naive", m = 7, alpha = 0., beta = 1e4,\
                                              rho = 1e-2, fix_beta = True, beta_max = 1e6, nu = 1e4,\
                                              lb = 0, ub = 1e4, maxiter = 1e4, abstol = 1e-6,\
                                              reltol = 1e-4, eigtol = 1e-9,\
                                              record_weights = False, record_objective = False, verbose = True):
  if is_data_matrix or S.shape[0] != S.shape[1]:
    A = build_initial_graph(S, m = m)
    D =  np.diag(.5 * (np.sum(A,axis=1) + np.sum(A,axis=0)))
    L = D - .5 * (A + A.T)
    S = np.linalg.pinv(L)
    is_data_matrix = True
  # number of nodes
  n = S.shape[0]
  # note now that S is always some sort of similarity matrix
  J = np.ones([n,n])/n
  # l1-norm penalty factor
  H = alpha * (2 * np.eye(n) - np.ones([n, n]))
  K = S + H
  # compute initial guess
  if is_data_matrix:
    Sinv = L
  else:
    Sinv = np.linalg.pinv(S)
  # if w0 is either "naive" or "qp", compute it, else return w0
  w0 = w_init(w0, Sinv)
  # compute quantities on the initial guess
  Aw0 = Ad(w0)
  Lw0 = La(w0)
  V0 = joint_V_update(Aw0, z)
  psi0 = bipartite_psi_update(V0, Aw0)
  U0 = joint_U_update(Lw0, k)
  lambda0 = laplacian_lambda_update(lb, ub, beta, U0, Lw0, k)
  beta_seq = [beta]
  time_seq = [0]
  start_time = time()
  for i in np.arange(maxiter):
    w = joint_w_update(w0, Lw0, Aw0, U0, V0, lambda0, psi0, beta, nu, K)
    Lw = La(w)
    Aw = Ad(w)
    U = joint_U_update(Lw, k)
    V = joint_V_update(Aw, z)
    lambd = laplacian_lambda_update(lb, ub, beta, U, Lw, k)
    psi = bipartite_psi_update(V, Aw)
    time_seq.append(time()-start_time)
    werr = abs(w0 - w)
    has_w_converged = (np.all(werr <= .5 * reltol * (w + w0)) or np.all(werr <= abstol))
    time_seq.append(time()-start_time)
    eigvals = np.linalg.eigh(Lw)[0]
    if not(fix_beta):
      n_zero_eigenvalues = sum(abs(eigvals) < eigtol)
      if (k < n_zero_eigenvalues):
        beta = (1 + rho) * beta
      elif (k > n_zero_eigenvalues):
        beta = beta / (1 + rho)
      if (beta > beta_max):
        beta = beta_max
      beta_seq.append(beta)
    if (has_w_converged):
      break
    # update estimates
    w0 = w
    U0 = U
    V0 = V
    lambda0 = lambd
    psi0 = psi
    Lw0 = Lw
    Aw0 = Aw
  results = {"Laplacian" : Lw, "Adjacency" : Aw, "w" : w, "psi" : psi,
                  "lambd" : lambd, "V" : V, "U" : U, "elapsed_time" : time_seq,
                  "beta_seq" : beta_seq, "convergence" : has_w_converged}
  return(results)

def nb_connected_component(L):
    return np.sum(np.linalg.eigh(L)[0]<10**-12)

def is_bipartite(A):
    n=A.shape[0]
    co=[-1]*n
    def parc(u):
        for u in range(A.shape[0]):
            if A[u][v]>0:
                if co[v]==-1:
                    co[v]=1-co[u]
                    if not(parc(v)):
                        return False
                elif co[v]+co[u]!=1:
                    return False
        return True
    for u in range(n):
        if co[u]==-1:
            co[u]=0
            if not(parc(u)):
                return False
    return True

#print(learn_bipartite_k_component_graph(np.eye(3))["Laplacian"])
#print(learn_bipartite_graph(np.eye(3))["Laplacian"])
#testing functions

size_matrix = 100
l = np.ones([size_matrix*2, size_matrix*2])*0.1
l[:size_matrix, :size_matrix] = np.ones([size_matrix, size_matrix])*0.9
l[size_matrix:, size_matrix:] = np.ones([size_matrix, size_matrix])*0.9
l = l + np.eye(2*size_matrix)*0.1

n_samples = 10000
S = np.random.multivariate_normal(np.zeros(2*size_matrix), l, size=n_samples).T
di=learn_k_component_graph(S, k=5, is_data_matrix=True, maxiter=10**3, m=5,beta=10**0,lb=10-4)
L=di["Laplacian"]
print(di["convergence"])
print(L)
print(np.diagonal(L))
print(nb_connected_component(L))