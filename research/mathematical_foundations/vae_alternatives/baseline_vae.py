# baseline_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian

class SmoothVAE(nn.Module):
    """
    Proposition 2.1 の条件 (ii) を保証するため tanh を採用。
    潜在次元 d と観測次元 n を指定。
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),    nn.Tanh(),
        )
        self.fc_mu     = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),     nn.Tanh(),
            nn.Linear(hidden, input_dim),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decode(z), mu, logvar


def vae_elbo(x_recon, x, mu, logvar, beta=1.0):
    recon = F.mse_loss(x_recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl


def train_vae(model, X, epochs=500, lr=1e-3, device="cpu"):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = X.to(device)
    for ep in range(epochs):
        opt.zero_grad()
        x_recon, mu, logvar = model(X)
        loss = vae_elbo(x_recon, X, mu, logvar)
        loss.backward()
        opt.step()
        if (ep + 1) % 100 == 0:
            print(f"[VAE] epoch {ep+1:4d} | loss = {loss.item():.2f}")
    return model


def pullback_metric(decoder, z):
    """
    g_{ij}(z) = J^T J at a single point z (shape: (d,)).
    Returns (d, d) torch.Tensor.
    """
    z = z.detach().requires_grad_(True)
    J = jacobian(decoder, z, create_graph=False)   # (input_dim, d)
    return J.T @ J


def check_proposition_2_1(decoder, Z_samples, sigma_threshold=1e-4):
    """
    Proposition 2.1 の条件 (iii) の数値検証。
    Z_samples: (N, d) 潜在空間サンプル点。
    """
    sigmas_min = []
    for z in Z_samples:
        J = jacobian(decoder, z, create_graph=False)
        S = torch.linalg.svdvals(J)
        sigmas_min.append(S.min().item())
    sigmas_min = torch.tensor(sigmas_min)
    passed = (sigmas_min > sigma_threshold).all().item()
    return {
        "passed": bool(passed),
        "sigma_min_global": sigmas_min.min().item(),
        "sigma_min_mean": sigmas_min.mean().item(),
        "n_violations": int((sigmas_min <= sigma_threshold).sum().item()),
    }


if __name__ == "__main__":
    from toy_data import make_swiss_roll, to_torch
    X_np, _ = make_swiss_roll(n_samples=1000, seed=0)
    X = to_torch(X_np)
    vae = SmoothVAE(input_dim=3, latent_dim=2, hidden=64)
    train_vae(vae, X, epochs=500)

    # ランク検証
    with torch.no_grad():
        mu, _ = vae.encode(X[:100])
    Z = mu.detach()
    report = check_proposition_2_1(vae.decode, Z)
    print("Proposition 2.1 (iii):", report)

    # 1 点での pullback 計量
    z0 = Z[0]
    g = pullback_metric(vae.decode, z0)
    print("g at z_0:\n", g)
    print("eigenvalues:", torch.linalg.eigvalsh(g))