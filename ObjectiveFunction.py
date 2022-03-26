import numpy as np

def laplacian_objectiveFunction(Lw, U, lambd, K, beta):
  return laplacian_likelihood(Lw, lambd, K) +laplacian_prior(beta, Lw, lambd, U)

def laplacian_likelihood(Lw,lambd,K):
    return np.sum(-np.log(lambd))+np.sum(np.diagonal(K@Lw))

def laplacian_prior(beta,Lw,lambd,U):
    to_cross = lambd**0.5 * U.T
    return 0.5 * beta * np.linalg.norm(Lw - (to_cros.T @ to_cross)**2)

def bipartite_obj_fun(Aw,Lw,V,psi,K,J,nu):
    return bipartite_likelihood(Lw=Lw,K=K,J=K)+bipartite_prior(nu=nu,Aw=Aw,psi=psi,V=V)

def bipartite_likelihood(LW,K,J):
    return np.sum(-np.log(np.linalg.eigh(Lw+J)[0])+np.diagonal(K@Lw))

def bipartite_prior(nu, Aw, psi, V):
  return 0.5 * nu * np.linalg.norm(Aw - V @ np.diag(psi) @ V.T**2)



def joint_obj_fun(Lw, Aw, U, V, lambd, psi, beta, nu, K):
  return  laplacian_likelihood(Lw = Lw, lambd = lambd, K = K) +\
         joint_prior(beta = beta, nu = nu, Lw = Lw, Aw = Aw, U = U, V = V,\
                     lambd = lambd, psi = psi)

"""
joint.likelihood <- function(...) {
  return(laplacian.likelihood(...))
}
"""

def joint_prior(beta, nu, Lw, Aw, U, V, lambd, psi):
  return laplacian_prior(beta = beta, Lw = Lw, lambd = lambd, U = U) +\
         bipartite_prior(nu = nu, Aw = Aw, psi = psi, V = V)

def vanilla_objective(Theta, K):
  p = Theta.shape[0]
  return np.sum(np.diagonal(Theta @ K)) - np.sum(np.log(np.linalg.eigh(Theta)[0][1:]))
