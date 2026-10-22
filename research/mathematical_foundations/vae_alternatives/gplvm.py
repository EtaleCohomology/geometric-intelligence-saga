# gplvm.py
import torch
import numpy as np
import gpytorch
from gpytorch.models.gplvm.bayesian_gplvm import BayesianGPLVM
from gpytorch.models.gplvm.latent_variable import VariationalLatentVariable
from gpytorch.means import ZeroMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.priors import NormalPrior


class MyGPLVM(BayesianGPLVM):
    def __init__(self, n, data_dim, latent_dim, n_inducing=25):
        batch_shape = torch.Size([data_dim])
        inducing_inputs = torch.randn(data_dim, n_inducing, latent_dim)
        q_f = CholeskyVariationalDistribution(n_inducing, batch_shape=batch_shape)
        q_u = VariationalStrategy(self, inducing_inputs, q_f, learn_inducing_locations=True)
        X_init = torch.nn.Parameter(torch.randn(n, latent_dim))
        prior_x = NormalPrior(torch.zeros(n, latent_dim), torch.ones(n, latent_dim))
        X = VariationalLatentVariable(n, data_dim, latent_dim, X_init, prior_x)
        super().__init__(X, q_u)
        self.mean_module = ZeroMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(RBFKernel(batch_shape=batch_shape, ard_num_dims=latent_dim),
                                         batch_shape=batch_shape)

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gplvm(Y, latent_dim=2, n_iter=500, lr=0.01):
    Y = torch.tensor(Y, dtype=torch.float32)
    n, data_dim = Y.shape
    model = MyGPLVM(n, data_dim, latent_dim)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([data_dim]))
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n)
    opt = torch.optim.Adam([
        {"params": model.parameters()},
        {"params": likelihood.parameters()},
    ], lr=lr)
    model.train(); likelihood.train()
    for it in range(n_iter):
        opt.zero_grad()
        sample = model.sample_latent_variable()
        output = model(sample)
        loss = -mll(output, Y.T).sum()
        loss.backward()
        opt.step()
        if (it + 1) % 100 == 0:
            print(f"[GPLVM] iter {it+1:4d} | loss = {loss.item():.2f}")
    return model, likelihood


def analytic_pullback_rbf(model, Z):
    """
    RBF カーネルの場合、pullback 計量の期待値は解析的 (Tosi et al. 2014).
    E[g_ij(z)] = sum_d sigma_f^2 * (1/ell_i ell_j) * E[f_d(z) gradient terms]
    ここでは簡略化: 事後分散を使って数値的に近似.
    """
    # 実装簡略: deterministic posterior mean のヤコビアンを使う
    from torch.autograd.functional import jacobian
    def mean_fn(z):
        Z_batch = z.unsqueeze(0)  # (1, latent_dim)
        model.eval()
        with torch.no_grad(): pass
        # Posterior mean at Z_batch
        # Note: gpytorch の BayesianGPLVM はこの用途には追加実装が必要
        raise NotImplementedError("gpytorch BayesianGPLVM の posterior mean API 要整備")

if __name__ == "__main__":
    from toy_data import make_swiss_roll
    X, _ = make_swiss_roll(n_samples=200)
    model, lik = train_gplvm(X, latent_dim=2, n_iter=300)
    print("Trained GPLVM. Latent locations shape:",
          model.X.q_mu.shape if hasattr(model.X, "q_mu") else "n/a")