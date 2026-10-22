# info_geometry.py
import numpy as np
import torch

def fisher_metric_mvn(mean_fn, cov_fn, theta, eps=1e-4):
    """
    多変量ガウシアン族 N(mu(theta), Sigma(theta)) の Fisher 計量を有限差分で。
    
    Args:
        mean_fn:  theta -> mu (torch.Tensor, shape=(D,))
        cov_fn:   theta -> Sigma (torch.Tensor, shape=(D, D))
        theta:    (d,) torch.Tensor
    
    Returns:
        g: (d, d) Fisher 計量
    """
    d = theta.shape[0]
    theta = theta.detach().clone().requires_grad_(True)
    mu = mean_fn(theta)
    Sigma = cov_fn(theta)
    Sigma_inv = torch.linalg.inv(Sigma)

    # d mu / d theta^i
    dmu = torch.stack([
        torch.autograd.grad(mu.sum(), theta, create_graph=True, retain_graph=True)[0]
        if False else _finite_diff_vec(mean_fn, theta, i, eps)
        for i in range(d)
    ])  # (d, D)

    # d Sigma / d theta^i
    dSigma = torch.stack([
        _finite_diff_mat(cov_fn, theta, i, eps)
        for i in range(d)
    ])  # (d, D, D)

    g = torch.zeros(d, d)
    for i in range(d):
        for j in range(d):
            # mean term
            term_mean = dmu[i] @ Sigma_inv @ dmu[j]
            # covariance term
            M = Sigma_inv @ dSigma[i] @ Sigma_inv @ dSigma[j]
            term_cov = 0.5 * torch.trace(M)
            g[i, j] = term_mean + term_cov
    return g


def _finite_diff_vec(fn, theta, i, eps):
    t_plus = theta.clone(); t_plus[i] += eps
    t_minus = theta.clone(); t_minus[i] -= eps
    return (fn(t_plus) - fn(t_minus)) / (2 * eps)


def _finite_diff_mat(fn, theta, i, eps):
    return _finite_diff_vec(fn, theta, i, eps)


if __name__ == "__main__":
    # toy: 2D ガウシアン族、theta = (mu_1, mu_2)
    def mean_fn(theta):
        return theta  # mu = theta
    def cov_fn(theta):
        # 共分散が位置依存: Sigma = I + 0.1 * theta[0]^2 * I
        return (1.0 + 0.1 * theta[0]**2) * torch.eye(2)

    theta = torch.tensor([1.0, -0.5])
    g = fisher_metric_mvn(mean_fn, cov_fn, theta)
    print("Fisher metric at theta = (1, -0.5):\n", g)
    print("eigenvalues:", torch.linalg.eigvalsh(g))