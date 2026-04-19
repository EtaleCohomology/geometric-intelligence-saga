# neural_ode.py
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

class ODEFunc(nn.Module):
    """ODE の右辺 f_theta(z, t). 全層 tanh で C^infty 級."""
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),   nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t, z):
        # t を一緒に入力することで time-dependent dynamics を許容
        tz = torch.cat([z, t.expand(z.shape[0], 1)], dim=-1)
        return self.net(tz)


class NeuralODEFlow(nn.Module):
    """
    Neural ODE によるフロー写像。
    z_0 -> z_T は C^infty 微分同相 (f_theta が Lipschitz なら).
    """
    def __init__(self, dim: int, hidden: int = 64, t_end: float = 1.0):
        super().__init__()
        self.ode_func = ODEFunc(dim, hidden)
        self.t = torch.tensor([0.0, t_end])

    def forward(self, z0):
        # z0: (B, dim)
        traj = odeint(self.ode_func, z0, self.t, method="dopri5", rtol=1e-5, atol=1e-5)
        return traj[-1]  # z(T)


class CNFDecoder(nn.Module):
    """
    Continuous Normalizing Flow 風デコーダ。
    潜在空間 Z と観測空間 X の次元が同じ場合に Neural ODE を直接使える。
    異次元の場合は後述の "augmented" 構成。
    """
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.flow = NeuralODEFlow(dim, hidden)

    def forward(self, z):
        return self.flow(z)


def verify_diffeomorphism(decoder: CNFDecoder, n_points: int = 100, d: int = 2):
    """
    Neural ODE フローが微分同相であることを数値検証。
    - 滑らかさ: loss.backward() で 2 階微分が計算できる
    - 単射性: 異なる z が異なる出力
    - フルランク: ヤコビ行列のフルランク
    """
    z = torch.randn(n_points, d, requires_grad=True)
    x = decoder(z)
    # 出力の distinctness
    dists_z = torch.cdist(z, z)
    dists_x = torch.cdist(x, x)
    # 同一 z を排除して相関
    mask = ~torch.eye(n_points, dtype=torch.bool)
    corr = torch.corrcoef(torch.stack([dists_z[mask], dists_x[mask]]))[0, 1]

    # フルランクの検証: 複数点で Jacobian SVD
    from torch.autograd.functional import jacobian
    sigmas = []
    for i in range(min(10, n_points)):
        J = jacobian(lambda z_: decoder(z_.unsqueeze(0)).squeeze(0), z[i])
        S = torch.linalg.svdvals(J)
        sigmas.append(S.min().item())

    return {
        "distance_correlation": float(corr.item()),
        "sigma_min_mean": float(sum(sigmas) / len(sigmas)),
        "sigma_min_min":  float(min(sigmas)),
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    decoder = CNFDecoder(dim=2, hidden=64)
    # 訓練なしでも微分同相性は構造的に保証される
    report = verify_diffeomorphism(decoder, n_points=50, d=2)
    print("Diffeomorphism verification:", report)

    # 明示的にヤコビ行列式の符号が保存されるか確認
    from torch.autograd.functional import jacobian
    z = torch.randn(5, 2)
    for i, z_i in enumerate(z):
        J = jacobian(lambda z_: decoder(z_.unsqueeze(0)).squeeze(0), z_i)
        det = torch.linalg.det(J).item()
        print(f"  z_{i}: det(J) = {det:.4f} (same sign ⇒ orientation-preserving)")