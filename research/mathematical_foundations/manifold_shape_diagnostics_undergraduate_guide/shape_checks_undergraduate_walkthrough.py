"""Undergraduate walkthrough: hearing the shape of a GI map through its eigenvalues.

Companion to README.md in this folder (a plain-language guide in Japanese).
The full research version is ../manifold_shape_diagnostics/shape_checks.py.

Run:  python shape_checks_undergraduate_walkthrough.py
Needs numpy, scipy, scikit-learn. CPU only, about 10 seconds.

Step 1  tiny graphs by hand: the number of zero eigenvalues = the number of pieces
Step 2  points on a circle: the graph eigenvalues come close to 0, 1, 1, 4, 4 (= k^2),
        but only when the neighbourhood is wide enough
Step 3  a cylinder and three maps: correct, torn open, folded
        The torn map has zero reconstruction error, yet its eigenvalues differ.
"""
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

RNG = np.random.default_rng(2026)


# ------------------------------------------------------------------ Step 1
def step1_tiny_graphs():
    print("Step 1: tiny graphs")
    # Path graph  1 - 2 - 3.  L = D - W (degree matrix minus adjacency matrix)
    W = np.array([[0, 1, 0],
                  [1, 0, 1],
                  [0, 1, 0]], dtype=float)
    L = np.diag(W.sum(axis=1)) - W
    print("  path 1-2-3, L =\n", L)
    print("  eigenvalues:", np.round(np.linalg.eigvalsh(L), 6), " (one zero: one piece)")

    # Two separate edges  1 - 2   3 - 4
    W = np.zeros((4, 4))
    W[0, 1] = W[1, 0] = 1
    W[2, 3] = W[3, 2] = 1
    L = np.diag(W.sum(axis=1)) - W
    print("  two separate edges, eigenvalues:", np.round(np.linalg.eigvalsh(L), 6),
          " (two zeros: two pieces)")


# ------------------------------------------------------------------ shared tools
def knn_mask(Z, k):
    """Connect each point to its k nearest neighbours (measured in the coordinates Z)."""
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)
    rows = np.repeat(np.arange(len(Z)), k)
    A = csr_matrix((np.ones(len(rows)), (rows, idx[:, 1:].ravel())), shape=(len(Z), len(Z)))
    return ((A + A.T) > 0).astype(float)


def laplace_eigenvalues(mask, X, eps, n_eig=6):
    """Smallest eigenvalues of a graph approximation of the Laplace-Beltrami operator.

    Edge weight exp(-|x_i - x_j|^2 / eps), density normalisation of Coifman-Lafon (alpha = 1),
    then (1 - eigenvalue of the Markov matrix) * 4 / eps.
    """
    M = mask.tocoo()
    K = np.zeros((len(X), len(X)))
    K[M.row, M.col] = np.exp(-np.sum((X[M.row] - X[M.col]) ** 2, axis=1) / eps)
    K += np.eye(len(X))
    q = K.sum(axis=1)
    K = K / np.outer(q, q)                 # remove the effect of uneven sampling density
    d = K.sum(axis=1)
    S = K / np.sqrt(np.outer(d, d))        # symmetric matrix with the same eigenvalues as the Markov matrix
    w = np.sort(eigh(S, eigvals_only=True))[::-1][:n_eig]
    return (1.0 - w) * 4.0 / eps


# ------------------------------------------------------------------ Step 2
def step2_circle(n=800, eps=0.01):
    print("Step 2: points on a circle of radius 1")
    th = RNG.uniform(0, 2 * np.pi, n)
    X = np.column_stack([np.cos(th), np.sin(th)])
    print("  exact (k^2)              : [0. 1. 1. 4. 4.]")
    for k in (100, 40):
        lam = laplace_eigenvalues(knn_mask(X, k), X, eps, 5)
        print(f"  graph eigenvalues, k={k:3d} :", np.round(lam, 2))
    print("  -> with too few neighbours the Gaussian weight is cut off and the eigenvalues come out small.")
    print("     The graph construction changes the answer, so fix it before looking at results.")


# ------------------------------------------------------------------ Step 3
def step3_cylinder(n=1500, k=200, eps=0.05):
    print("Step 3: a cylinder (circle of radius 1) x (height 0 to 3), and three maps")
    th = RNG.uniform(0, 2 * np.pi, n)
    z = RNG.uniform(0, 3, n)
    X = np.column_stack([np.cos(th), np.sin(th), z])
    lam_data = laplace_eigenvalues(knn_mask(X, k), X, eps)
    exact = np.sort([a ** 2 + (np.pi * b / 3) ** 2 for a in range(-3, 4) for b in range(4)])[:6]
    print("  exact eigenvalues of the cylinder:", np.round(exact, 2))
    print("  data side (graph)               :", np.round(lam_data, 2))

    zf = 1.5 + np.abs(z - 1.5)             # the folded map cannot tell z from 3 - z
    maps = {
        # name: (where each point goes on the map, the point the decoder gives back)
        "correct": (np.column_stack([(1 + z / 3) * np.cos(th), (1 + z / 3) * np.sin(th)]), X),
        "torn open": (np.column_stack([th, z]), X),
        "folded": (np.column_stack([(zf / 1.5) * np.cos(th), (zf / 1.5) * np.sin(th)]),
                   np.column_stack([np.cos(th), np.sin(th), zf])),
    }
    for name, (Z, X_back) in maps.items():
        lam = laplace_eigenvalues(knn_mask(Z, k), X_back, eps)
        rec = np.sqrt(((X_back - X) ** 2).sum(axis=1)).mean()
        gap = np.abs(lam - lam_data).max()
        print(f"  {name:10s} reconstruction error {rec:.3f}, eigenvalues {np.round(lam, 2)}, "
              f"largest gap {gap:.3f}")
    print("  -> 'torn open' has zero reconstruction error, but a new small eigenvalue near 0.25 appears.")


if __name__ == "__main__":
    step1_tiny_graphs()
    step2_circle()
    step3_cylinder()
