from utilscc import*
import numpy as np
from ObjectiveFunction import*
def learn_k_component_graph (S, is_data_matrix = FALSE, k = 1, w0 = "naive", lb = 0, ub = 1e4, alpha = 0,\
                                    beta = 1e4, beta_max = 1e6, fix_beta = TRUE, rho = 1e-2, m = 7,\
                                    maxiter = 1e4, abstol = 1e-6, reltol = 1e-4, eigtol = 1e-9,\
                                    record_objective = False, record_weights = False, verbose = True):
  if (is_data_matrix || ncol(S) != nrow(S)) {
    A <- build_initial_graph(S, m = m)
    D <- diag(.5 * colSums(A + t(A)))
    L <- D - .5 * (A + t(A))
    S <- MASS::ginv(L)
    is_data_matrix <- TRUE
  }
  # number of nodes
  n <- nrow(S)
  # l1-norm penalty factor
  H <- alpha * (2 * diag(n) - matrix(1, n, n))
  K <- S + H
  # find an appropriate inital guess
  if (is_data_matrix)
    Sinv <- L
  else
    Sinv <- MASS::ginv(S)
  # if w0 is either "naive" or "qp", compute it, else return w0
  w0 <- w_init(w0, Sinv)
  # compute quantities on the initial guess
  Lw0 <- L(w0)
  U0 <- laplacian.U_update(Lw = Lw0, k = k)
  lambda0 <- laplacian.lambda_update(lb = lb, ub = ub, beta = beta, U = U0,
                                     Lw = Lw0, k = k)
  # save objective function value at initial guess
  if (record_objective) {
    ll0 <- laplacian.likelihood(Lw = Lw0, lambda = lambda0, K = K)
    fun0 <- ll0 + laplacian.prior(beta = beta, Lw = Lw0, lambda = lambda0, U = U0)
    fun_seq <- c(fun0)
    ll_seq <- c(ll0)
  }
  beta_seq <- c(beta)
  if (record_weights)
    w_seq <- list(Matrix::Matrix(w0, sparse = TRUE))
  time_seq <- c(0)
  if (verbose)
    pb <- progress::progress_bar$new(format = "<:bar> :current/:total  eta: :eta  beta: :beta  kth_eigval: :kth_eigval relerr: :relerr",
                                     total = maxiter, clear = FALSE, width = 120)
  start_time <- proc.time()[3]
  for (i in 1:maxiter) {
    w <- laplacian.w_update(w = w0, Lw = Lw0, U = U0, beta = beta,
                            lambda = lambda0, K = K)
    Lw <- L(w)
    U <- laplacian.U_update(Lw = Lw, k = k)
    lambda <- laplacian.lambda_update(lb = lb, ub = ub, beta = beta, U = U,
                                      Lw = Lw, k = k)
    # compute negloglikelihood and objective function values
    if (record_objective) {
      ll <- laplacian.likelihood(Lw = Lw, lambda = lambda, K = K)
      fun <- ll + laplacian.prior(beta = beta, Lw = Lw, lambda = lambda, U = U)
      ll_seq <- c(ll_seq, ll)
      fun_seq <- c(fun_seq, fun)
    }
    if (record_weights)
      w_seq <- rlist::list.append(w_seq, Matrix::Matrix(w, sparse = TRUE))
    # check for convergence
    werr <- abs(w0 - w)
    has_w_converged <- all(werr <= .5 * reltol * (w + w0)) || all(werr <= abstol)
    time_seq <- c(time_seq, proc.time()[3] - start_time)
    if (verbose || (!fix_beta)) eigvals <- eigval_sym(Lw)
    if (verbose) {
      pb$tick(token = list(beta = beta, kth_eigval = eigvals[k],
                           relerr = 2 * max(werr / (w + w0), na.rm = 'ignore')))
    }
    if (!fix_beta) {
      n_zero_eigenvalues <- sum(abs(eigvals) < eigtol)
      if (k <= n_zero_eigenvalues)
        beta <- (1 + rho) * beta
      else if (k > n_zero_eigenvalues)
        beta <- beta / (1 + rho)
      if (beta > beta_max)
        beta <- beta_max
      beta_seq <- c(beta_seq, beta)
    }
    if (has_w_converged)
      break
    # update estimates
    w0 <- w
    U0 <- U
    lambda0 <- lambda
    Lw0 <- Lw
  }
  # compute the adjancency matrix
  Aw <- A(w)
  results <- list(Laplacian = Lw, Adjacency = Aw, w = w, lambda = lambda, U = U,
                  elapsed_time = time_seq, convergence = has_w_converged,
                  beta_seq = beta_seq)
  if (record_objective) {
    results$obj_fun <- fun_seq
    results$loglike <- ll_seq
  }
  if (record_weights)
    results$w_seq <- w_seq
  return(results)
}
