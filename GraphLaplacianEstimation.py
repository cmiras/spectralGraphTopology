import numpy as np
from utilscc import*
from ObjectiveFunction import*
from operators import*
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
  mask = Ainv(A_mask) > 0
  w = w_init("naive", Sinv)[mask]
  wk = w
  # number of nonzero edges
  m = sum(mask)#.5 * sum(A_mask > 0)
  # l1-norm penalty factor
  J = np.ones(p,p)/p
  H = 2 * np.eye(p) - p * J
  K = S + alpha * H
  E = get_incidence_from_adjacency(A_mask)
  R = t(E) @ K @ E
  r = R.shape[0]
  G = [E, np.arange(p)]
  assert False, "à vérifier"
  if record_objective:
    z = np.zeros((p*(p-1))//2)
    z[mask] = wk
    fun = vanilla_objective(La(z), K)
  for k in np.arange(maxiter):
    w_aug =np.concatenate(wk,np.array([1/p]))
    G_aug_t = G.T * w_aug
    G_aug = G_aug_t.T
    Q = G_aug_t @ np.linalg.lstsq(G_aug @ G.T, G_aug)
    Q = Q[:m, :m]
    wk = (np.diagonal(Q) / np.diagonal(R))**0.5
    has_converged = np.linalg.norm(w - wk, "2") / np.linalg.norm(w, "2") < reltol
    if (has_converged and k > 1):
        break
    w = wk
  z = np.zeros((p*(p-1))//2)
  z[mask] = wk
  results = {"Laplacian" : La(z), "Adjacency" : Ad(z), "maxiter" : k,"convergence" : has_converged}
  return results


def obj_func(E, K, w, J):
  p = J.shpae[1]
  EWEt = E @ np.diag(w) @ t(E)
  Gamma = EWEt + J
  lambd = np.linalg.eigh(Gamma)[0][1:p]
  return np.sum(np.diagonal(E @ np.diag(w) @ E.T @ K)) - np.sum(np.log(lambd))

#' @title Learn the weighted Laplacian matrix of a graph using the ADMM method
#'
#' @param S a pxp sample covariance/correlation matrix
#' @param A_mask the binary adjacency matrix of the graph
#' @param alpha L1 regularization hyperparameter
#' @param rho ADMM convergence rate hyperparameter
#' @param maxiter the maximum number of iterations
#' @param reltol relative tolerance on the Laplacian matrix estimation
#' @param record_objective whether or not to record the objective function. Default is FALSE
#' @param verbose if TRUE, then a progress bar will be displayed in the console. Default is TRUE
#' @return A list containing possibly the following elements:
#' \item{\code{Laplacian}}{the estimated Laplacian Matrix}
#' \item{\code{Adjacency}}{the estimated Adjacency Matrix}
#' \item{\code{convergence}}{boolean flag to indicate whether or not the optimization converged}
#' \item{\code{obj_fun}}{values of the objective function at every iteration in case record_objective = TRUE}
#' @author Ze Vinicius, Jiaxi Ying, and Daniel Palomar
#' @references Licheng Zhao, Yiwei Wang, Sandeep Kumar, and Daniel P. Palomar.
#'             Optimization Algorithms for Graph Laplacian Estimation via ADMM and MM.
#'             IEEE Trans. on Signal Processing, vol. 67, no. 16, pp. 4231-4244, Aug. 2019
#' @export
def learn_laplacian_gle_admm(S, A_mask = None, alpha = 0, rho = 1, maxiter = 10000,\
                                     reltol = 1e-5, record_objective = False, verbose = True):
  p = S.shape[0]
  if A_mask==None:
    A_mask = np.ones(p, p) - np.eye(p)
  Sinv = np.linalg.pinv(S)
  w = w_init("naive", Sinv)
  Theta = La(w)
  Yk = Theta
  Ck = Theta
  C = Theta
  # ADMM constants
  mu = 2
  tau = 2
  # l1-norm penalty factor
  J = np.ones(p,p) / p
  H = 2 * np.eye(p) - p * J
  K = S + alpha * H
  # ADMM loop
  assert False, "decomposition QR à voir demain"
  #P = qr.Q(qr(rep(1, p)), complete=TRUE)[, 2:p]
  """for k in np.arange(maxiter):
    Gamma = t(P) @ ((K + Yk) / rho - Ck) @ P
    lambd,U = np.linalg.eigh(Gamma)
    d = .5 * c(sqrt(lambd ^ 2 + 4 / rho) - lambda)
    Xik = crossprod(sqrt(d) * t(U))
    Thetak = P @ Xik @ t(P)
    Ck_tmp = Yk / rho + Thetak
    Ck = (diag(pmax(0, diag(Ck_tmp))) +
           A_mask * pmin(0, Ck_tmp))
    Rk = Thetak - Ck
    Yk = Yk + rho * Rk
    if (record_objective)
      fun = c(fun, vanilla.objective(Thetak, K))
    has_converged =  np.linalg.norm(Theta - Thetak) / np.linalg.norm(Theta) < reltol
    if (has_converged && k > 1) break
    s = rho * np.linalg.norm(C - Ck)
    r = np.linalg.norm(Rk, "F")
    if (r > mu * s)
      rho = rho * tau
    else if (s > mu * r)
      rho = rho / tau
    Theta = Thetak
    C = Ck
    if (verbose)
      pb$tick()
  }
  results = {"Laplacian" : Thetak, "Adjacency" : np.diag(np.diagonal(Thetak)) - Thetak,\
                  "convergence" : has_converged}

  return results"""


# compute the partial augmented Lagragian
def aug_lag(K, P, Xi, Y, C, d, rho):
  PXiPt = P @ Xi @ P.T
  return(np.sum(np.diagonal(Xi @ P.T @ K @ P)) - np.sum(np.log(d)) +\
         np.sum(np.diagonal(Y.T @ (PXiPt - C))) + .5 * rho * np.linalg.norm(PXiPt - C) ** 2)
