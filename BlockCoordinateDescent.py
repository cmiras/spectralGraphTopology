from qpsolvers import *
from operators import *
import numpy as np

def w_init(w0, Sinv):
  """
  Initialize w0, the vectorized upper triangular coefficients of the adjacency matrix
  """
  if type(w0) is str:
    if (w0 == "qp"):
      R = vecLmat(Sinv.shape[1])
      qp = 0
      assert False,"idk"
      #quadprog::solve.QP(crossprod(R), t(R) %*% vec(Sinv), diag(ncol(R)))
      w0 = qp#qp$solution
    elif (w0 == "naive"):
      w0 = Linv(Sinv)
      w0[w0 < 0] = 0 # Should not happen
  return w0



def laplacian_w_update(w, Lw, U, beta, lambd, K, p):
  """
  Update w according to equation 38
  """
  t = lambd[:, None]**0.5 * U.T
  c = Lstar(t.T@t - K / beta)
  grad_f = Lstar(Lw) - c
  if 1:
    M_grad_f = - Lstar(La(grad_f))
    wT_M_grad_f = np.sum(w * M_grad_f)
    dwT_M_dw = np.sum(grad_f * M_grad_f)
  # exact line search
    t = (wT_M_grad_f - np.sum(c * grad_f)) / dwT_M_dw
  else:
      t=1/(2*p)
  w_update = w - t * grad_f
  w_update[w_update < 0] = 0
  return w_update



def joint_w_update(w, Lw, Aw, U, V, lambd, psi, beta, nu, K):
  t=lambd[:, None]**0.5*U.T
  ULmdUT = t.T@t
  VPsiVT = V @ np.diag(psi) @ V.T
  c1 = Lstar(beta * ULmdUT - K)
  c2 = nu * Astar(VPsiVT)
  Mw = Lstar(Lw)
  Pw = 2 * w
  grad_f1 = beta * Mw - c1
  M_grad_f1 = Lstar(La(grad_f1))
  grad_f2 = nu * Pw - c2
  P_grad_f2 = 2 * grad_f2
  grad_f = grad_f1 + grad_f2
  t = np.sum((beta * Mw + nu * Pw - (c1 + c2)) * grad_f) / np.sum(grad_f * (beta * M_grad_f1 + nu * P_grad_f2))
  w_update = w - t * (grad_f1 + grad_f2)
  w_update[w_update < 0] = 0
  return w_update


def bipartite_w_update(w, Aw, V, nu, psi, K, J, Lips):
  reg_eps = 0
  grad_h = 2 * w - Astar(V @ np.diag(psi) @ V.T) #+ Lstar(K) / beta#
  w_update = w - (Lstar(np.linalg.inv(La(w) + J+np.eye(J.shape[0])*reg_eps) + K) + nu * grad_h) / (2 * nu + Lips)
  w_update[w_update < 0] = 0#TODO faire en sorte que la régularisation ligne précédent ne soit pas nécessaire
  return w_update



def laplacian_U_update(Lw, k):
  """
  Return all but the k first eigenvectors of the Laplacian Lw
  """
  return np.linalg.eigh(Lw)[1][:, k:]


def bipartite_V_update(Aw, z):
  n = Aw.shape[1]
  V = np.linalg.eigh(Aw)[1]
  return np.concatenate([V[:, :(n - z)//2], V[:,(n + z)//2:n]],axis=1)


def joint_U_update(Lw,k):
  return np.linalg.eigh(Lw)[1][:, k:]


def joint_V_update(Aw,z):
  return bipartite_V_update(Aw,z)



def laplacian_lambda_update(lb, ub, beta, U, Lw, k):
  """
  Update lambda according to algorithm 1
  """
  q = Lw.shape[1] - k
  d = np.diagonal(U.T @ Lw @ U)
  # unconstrained solution as initial point
  lambd = .5 * (d + (d**2 + 4 / beta)**0.5)
  eps = 1e-9
  condition_ub = np.array([(lambd[q-1] - ub) <= eps])
  condition_lb = np.array([(lambd[0] - lb) >= -eps])
  condition_ordered = (lambd[1:q] - lambd[0:(q-1)]) >= -eps
  condition = np.concatenate([condition_ub,\
                 condition_lb,\
                 condition_ordered])
  if np.all(condition):
    return lambd
  else:
    greater_ub = lambd > ub
    lesser_lb = lambd < lb
    lambd[greater_ub] = ub
    lambd[lesser_lb] = lb
  condition_ub = np.array([(lambd[q-1] - ub) <= eps])
  condition_lb = np.array([(lambd[0] - lb) >= -eps])
  condition_ordered = (lambd[1:q] - lambd[:(q-1)]) >= -eps
  condition = np.concatenate([condition_ub,\
                 condition_lb,\
                 condition_ordered])
  if np.all(condition):
    return (lambd)
  else:
    print(lambd)
    raise ValueError('eigenvalues are not in increasing order consider increasing the value of beta')


def bipartite_psi_update(V, Aw, lb = -np.inf, ub = np.inf):
  c = np.diagonal(V.T @ Aw @ V)
  n = c.shape[0]
  c_tilde = .5 * (c[(n//2):][::-1] - c[:(n//2)])
  x = isoreg(c_tilde[::-1])
  #x <- stats::isoreg(rev(c_tilde))$yf # R
  x = np.concatenate((-x[::-1], x))
  #x <- c(-rev(x), x) # R
  x[x < lb] = lb
  x[x > ub] = ub
  return x



"""joint.lambda_update <- function(...) {
  return(laplacian.lambda_update(...))
}


def joint_psi_update(...):
  return(bipartite.psi_update(...))
"""
