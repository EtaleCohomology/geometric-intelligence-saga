# neural_ode_control.py
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

class ControlledSystem(nn.Module):
    """
    dz/dt = F(z) + u(t) * B(z) の形の affine 制御系.
    Pontryagin 最大原理の下で最適 u*(t) を求める.
    """
    def __init__(self, F_net, B_net, u_net):
        super().__init__()
        self.F = F_net    # drift
        self.B = B_net    # control matrix
        self.u = u_net    # control policy u(t, z)

    def forward(self, t, z):
        F_val = self.F(z)
        B_val = self.B(z)
        u_val = self.u(torch.cat([z, t.expand(z.shape[0], 1)], dim=-1))
        # u_val: (B, m), B_val: (B, d, m) -> drift contribution (B, d)
        return F_val + torch.einsum("bdm,bm->bd", B_val, u_val)


def pontryagin_adjoint_loss(system, z0, z_target, cost_fn, t_span=(0.0, 1.0)):
    """
    シンプルな trajectory optimization: 
    end-point cost + running cost を最小化する u を学習.
    訓練で自動的に随伴変数 (costate) が計算される。
    """
    t = torch.tensor(list(t_span))
    traj = odeint(system, z0, t, method="dopri5")
    z_final = traj[-1]
    # End-point cost
    J_terminal = ((z_final - z_target) ** 2).sum()
    # Running cost (quadratic in control: 積分近似)
    J_running = 0.0
    n_eval = 10
    for k in range(n_eval):
        tk = torch.tensor(t_span[0] + k * (t_span[1] - t_span[0]) / n_eval)
        zk = odeint(system, z0, torch.tensor([t_span[0], tk.item()]))[-1]
        uk = system.u(torch.cat([zk, tk.expand(zk.shape[0], 1)], dim=-1))
        J_running = J_running + 0.01 * (uk ** 2).sum() / n_eval
    return J_terminal + J_running