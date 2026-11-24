"""Global shape diagnostics for a GI map (data manifold vs decoder image).

Minimal working example for the research note README.md in this folder.
No external data. Runs in a few seconds on a laptop (CPU only).

The four checks compare two graphs built on the SAME points:
  - data side : edges and weights from distances in the (pre-standardised) data space
  - map side  : edges from neighbours in the latent space, weights from distances
                between the decoded points (a discrete stand-in for the pullback metric)

  Check 4  islands       : multiplicity of eigenvalue 0 of the graph Laplacian
                           (= number of connected components; exact for any graph)
  Check 5  spectrum      : low end of the density-normalised (Coifman-Lafon, alpha = 1)
                           diffusion Laplacian, same graph construction on both sides
  Check 6  heat kernel   : heat kernel signature per point, HKS(x, t) = sum_k exp(-lam_k t) phi_k(x)^2,
                           which is invariant to the sign and rotation ambiguity of eigenvectors
  Check 7  dimension     : Weyl-law slope of the eigenvalue counting function, N(lam) ~ C lam^(m/2)

Known-answer tests: data on a cylinder; a correct map (annulus latent), a torn map
(cylinder cut open into a rectangle) and a folded map (non-injective).
The checks should raise a yellow flag on the torn and folded maps and not on the correct map.

Reporting language: "no discrepancy found" or "yellow flag". Never "pass".
"""
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, laplacian
from sklearn.neighbors import NearestNeighbors

RNG = np.random.default_rng(2026)


# ---------------------------------------------------------------- graphs
def knn_mask(Z, k):
    """Symmetric k-nearest-neighbour adjacency (0/1) built in the coordinates Z."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    _, idx = nn.kneighbors(Z)
    rows = np.repeat(np.arange(len(Z)), k)
    cols = idx[:, 1:].ravel()
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(Z), len(Z)))
    return ((A + A.T) > 0).astype(float)


def diffusion_operator(mask, X, eps):
    """Coifman-Lafon alpha = 1 normalised kernel, returned in symmetric form S.

    mask : which pairs are connected (decided in the data space or in the latent space)
    X    : points used to measure edge weights (data points, or decoded points)
    The generator approximation of the Laplace-Beltrami operator is (I - P) * 4 / eps,
    where P is similar to S.
    """
    M = mask.tocoo()
    d2 = np.sum((X[M.row] - X[M.col]) ** 2, axis=1)
    K = np.zeros((len(X), len(X)))
    K[M.row, M.col] = np.exp(-d2 / eps)
    K += np.eye(len(X))
    q = K.sum(axis=1)
    K1 = K / np.outer(q, q)              # alpha = 1: remove the sampling density
    d = K1.sum(axis=1)
    S = K1 / np.sqrt(np.outer(d, d))     # symmetric conjugate of the Markov matrix
    return S, d


def low_spectrum(S, eps, n_eig):
    """Smallest n_eig eigenvalues of the Laplacian approximation and the eigenvectors of P."""
    w, V = eigh(S)
    order = np.argsort(-w)[:n_eig]       # largest eigenvalues of S = smallest of the Laplacian
    lam = (1.0 - w[order]) * 4.0 / eps
    return lam, V[:, order]


# ---------------------------------------------------------------- check 4
def count_islands(mask):
    L = laplacian(mask).toarray()
    ev = np.linalg.eigvalsh(L)
    n_zero = int((ev < 1e-9).sum())
    n_cc = connected_components(mask)[0]
    assert n_zero == n_cc, "dim ker L must equal the number of connected components"
    return n_zero


# ---------------------------------------------------------------- check 6
def heat_kernel_signature(lam, V, d, times):
    """HKS(x, t) = sum_k exp(-lam_k t) psi_k(x)^2 with psi the right eigenvectors of P.

    The eigenvectors of S are converted to those of P (psi = D^{-1/2} v) and normalised
    in L2(d). Summing psi_k^2 over each eigenspace removes the sign and rotation ambiguity.
    """
    psi = V / np.sqrt(d)[:, None]
    psi = psi / np.sqrt((psi ** 2 * d[:, None]).sum(axis=0))
    return np.stack([(np.exp(-lam * t) * psi ** 2).sum(axis=1) for t in times], axis=1)


def hks_agreement(h_data, h_map):
    """Mean over time scales of the rank correlation across points."""
    from scipy.stats import spearmanr
    return float(np.mean([spearmanr(h_data[:, j], h_map[:, j])[0] for j in range(h_data.shape[1])]))


# ---------------------------------------------------------------- check 7
def weyl_dimension(lam, lo=5, hi=40):
    """m ~ 2 * slope of log N(lam) against log lam, over a middle band of the spectrum.

    Only a rough indication: the top of a graph spectrum is unreliable and the band
    (lo, hi) changes the answer. Report it as an indication, not as an estimate.
    """
    lam = np.sort(lam[lam > 1e-9])
    idx = np.arange(lo, min(hi, len(lam)))
    slope = np.polyfit(np.log(lam[idx]), np.log(idx + 1), 1)[0]
    return 2.0 * slope


# ---------------------------------------------------------------- panel data
def unit_bootstrap(unit_ids, n_boot, rng=RNG):
    """Resample whole units (for example countries) with replacement.

    Rows of one unit (consecutive years of one country) are not independent, so the
    fluctuation of any spectral statistic should be estimated by resampling units,
    not rows. Yields index arrays.
    """
    units = np.unique(unit_ids)
    by_unit = {u: np.flatnonzero(unit_ids == u) for u in units}
    for _ in range(n_boot):
        pick = rng.choice(units, size=len(units), replace=True)
        yield np.concatenate([by_unit[u] for u in pick])


# ---------------------------------------------------------------- known-answer tests
def cylinder_data(n):
    th = RNG.uniform(0, 2 * np.pi, n)
    z = RNG.uniform(0, 3, n)
    X = np.column_stack([np.cos(th), np.sin(th), z])
    return th, z, X


def run_known_answer_tests(n=1500, k=200, eps=0.05, n_eig=40):
    th, z, X = cylinder_data(n)
    # (latent coordinates, decoded points). A perfect decoder returns the data points.
    # The folded encoder sends heights z and 3 - z to the same latent point, so no decoder
    # can tell them apart: the best it can do is decode both to the upper half.
    zf = 1.5 + np.abs(z - 1.5)
    latents = {
        "correct (annulus)": (np.column_stack([(1 + z / 3) * np.cos(th), (1 + z / 3) * np.sin(th)]), X),
        "torn (cut open)": (np.column_stack([th, z]), X),
        "folded (non-injective)": (np.column_stack([(zf / 1.5) * np.cos(th), (zf / 1.5) * np.sin(th)]),
                                   np.column_stack([np.cos(th), np.sin(th), zf])),
    }
    S_data, d_data = diffusion_operator(knn_mask(X, k), X, eps)
    lam_data, V_data = low_spectrum(S_data, eps, n_eig)
    times = [0.05, 0.1, 0.2, 0.5]
    h_data = heat_kernel_signature(lam_data, V_data, d_data, times)

    print(f"data side: islands = {count_islands(knn_mask(X, k))}, "
          f"Weyl indication of dimension = {weyl_dimension(lam_data):.2f} (true 2)")
    print("lowest eigenvalues, data side:", np.round(lam_data[:6], 2))
    results = {}
    for name, (Z, X_hat) in latents.items():
        mask = knn_mask(Z, k)
        S, d = diffusion_operator(mask, X_hat, eps)   # weights from the decoded points
        rec = float(np.sqrt(((X_hat - X) ** 2).sum(axis=1)).mean())
        lam, V = low_spectrum(S, eps, n_eig)
        gap = float(np.abs(lam[:6] - lam_data[:6]).max())
        hks = hks_agreement(h_data, heat_kernel_signature(lam, V, d, times))
        isl = count_islands(mask)
        results[name] = (gap, hks, isl)
        print(f"{name:24s} islands = {isl}, max gap of 6 lowest eigenvalues = {gap:.3f}, "
              f"HKS rank correlation = {hks:.3f}, mean reconstruction error = {rec:.3f}")
        print("   lowest eigenvalues:", np.round(lam[:6], 2))
    return results


def islands_example():
    """Two clusters in the data; a map that glues them must change the island count."""
    A = np.vstack([RNG.normal([0, 0], 0.3, (60, 2)), RNG.normal([4, 0], 0.3, (60, 2))])
    glued = A.copy()
    glued[:, 0] = np.where(glued[:, 0] > 2, glued[:, 0] - 4, glued[:, 0])   # map sends both clusters to one place
    print(f"islands: data side {count_islands(knn_mask(A, 6))}, gluing map side {count_islands(knn_mask(glued, 6))}")


if __name__ == "__main__":
    islands_example()
    res = run_known_answer_tests()
    g_ok = res["correct (annulus)"][0]
    for bad in ("torn (cut open)", "folded (non-injective)"):
        flag = "yellow flag" if res[bad][0] > 3 * g_ok else "no discrepancy found"
        print(f"{bad}: {flag}")
    print(f"correct (annulus): {'no discrepancy found' if g_ok < 0.1 else 'yellow flag'}")
