from qpsolvers import*
from operators import*
import numpy as np
def w_init(w0, Sinv):
  if type(wO) is string:
    if (w0 == "qp"):
      R = vecLmat(ncol(Sinv))
      qp <- assert False,idk#quadprog::solve.QP(crossprod(R), t(R) %*% vec(Sinv), diag(ncol(R)))
      w0 <- qp$solution
    elif (w0 == "naive"):
      w0 = Linv(Sinv)
      w0[w0 < 0] = 0
  return <o
}


def laplacian_w_update(w, Lw, U, beta, lambd, K):
  t = lambd**0.5 * U.T
  c = Lstar(t.T@t - K / beta)
  grad_f = Lstar(Lw) - c
  M_grad_f=- Lstar(La(grad_f))
  wT_M_grad_f = np.sum(w * M_grad_f)
  dwT_M_dw = np.sum(grad_f * M_grad_f)
  # exact line search
  t = (wT_M_grad_f - np.sum(c * grad_f)) / dwT_M_dw
  w_update = w - t * grad_f
  w_update[w_update < 0] = 0
  return w_update



def joint_w_update(w, Lw, Aw, U, V, lambd, psi, beta, nu, K):
  t=lambd**0.5*U.T
  ULmdUT = t.T@t
  VPsiVT = V @ np.diag(psi) @ V.T
  c1 = Lstar(beta * ULmdUT - K)
  c2 = nu * Astar(VPsiVT)
  Mw = Lstar(Lw)
  Pw = 2 * w
  grad_f1 = beta * Mw - c1
  M_grad_f1 = Lstar(L(grad_f1))
  grad_f2 = nu * Pw - c2
  P_grad_f2 = 2 * grad_f2
  grad_f = grad_f1 + grad_f2
  t = np.sum((beta * Mw + nu * Pw - (c1 + c2)) * grad_f) / np.sum(grad_f * (beta * M_grad_f1 + nu * P_grad_f2))
  w_update = w - t * (grad_f1 + grad_f2)
  w_update[w_update < 0] = 0
  return w_update


def bipartite_w_update(w, Aw, V, nu, psi, K, J, Lips):
  grad_h = 2 * w - Astar(V @ diag(psi) @ V.T) #+ Lstar(K) / beta
  w_update = w - (Lstar(np.linalg.inv(La(w) + J) + K) + nu * grad_h) / (2 * nu + Lips)
  w_update[w_update < 0] = 0
  return w_update



def laplacian_U_update(Lw, k):
  return np.linalg.eigh(Lw)[1][:, k:]


def bipartite_V_update(Aw, z):
  n = Aw.shape[1]
  V = np.linalg.eigh(Aw)[1]
  assert False, "j'ai pas compris"
  return(cbind(V[, 1:(.5*(n - z))], V[, (1 + .5*(n + z)):n]))
}


def joint_U_update(Lw,k):
  return np.linalg.eigh(Lw)[1][:, k:]


def joint_V_update(Aw,z):
  return bipartite_V_update(Aw,z)



def laplacian_lambda_update(lb, ub, beta, U, Lw, k):
  q = Lw.shape[1] - k
  d = np.diagonal(U.T @ Lw @ U)
  # unconstrained solution as initial point
  lambd = .5 * (d + (d^2 + 4 / beta)**0.5)
  eps = 1e-9
  condition = [lambd[q] - ub) <= eps,\
                 (lambd[0] - lb) >= -eps,\
                 (lambd[1:q] - lambd[0:(q-1)]) >= -eps]
  if min(condition):
    return lambd
  else:
    greater_ub = lambd > ub
    lesser_lb = lambd < lb
    lambd[greater_ub] = ub
    lambd[lesser_lb] = lb
  condition <- [(lambd[q] - ub) <= eps,\
                 (lambd[0] - lb) >= -eps,\
                 (lambd[1:q] - lambda[:(q-1)]) >= -eps)]
  if min(condition):
    return (lambd)
  else:
    print(lambd)
    raise ValueError('eigenvalues are not in increasing order consider increasing the value of beta')


def bipartite_psi_update(V, Aw, lb = -np.Inf, ub = np.Inf):
  c = np.diagonal(V.T @ Aw @ V)
  n = c.shape[0]
  c_tilde = .5 * (c[(n/2):][::-1] - c[:(n/2)])
  assert False, "jai pas compris"
  #x <- stats::isoreg(rev(c_tilde))$yf
  #x <- c(-, x)
  x[x < lb] = lb
  x[x > ub] = ub
  return x



joint.lambda_update <- function(...) {
  return(laplacian.lambda_update(...))
}


joint.psi_update <- function(...) {
  return(bipartite.psi_update(...))
}
