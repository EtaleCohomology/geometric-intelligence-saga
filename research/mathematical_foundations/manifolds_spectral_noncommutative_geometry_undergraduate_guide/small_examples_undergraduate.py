"""Small examples for the undergraduate guide README.md in this folder.

Run:  python small_examples_undergraduate.py      (numpy only, under a second)

Example 1  circle: eigenvalues of the Dirac-type operator -i d/dtheta are integers k,
           eigenvalues of the Laplacian are k^2 (checked with a finite-difference matrix)
Example 2  Weyl's law on the circle: N(lambda) is close to 2 * sqrt(lambda)
Example 3  two-point space: the Connes distance equals 1 / |m|
Example 4  spectral truncation: e^{i theta} and e^{-i theta} commute as functions,
           but their truncations to a few Fourier modes do not
"""
import numpy as np


def example1_circle(n=400):
    h = 2 * np.pi / n
    # periodic second-difference matrix: a discrete version of -d^2/dtheta^2 on the unit circle
    L = (2 * np.eye(n) - np.roll(np.eye(n), 1, axis=1) - np.roll(np.eye(n), -1, axis=1)) / h ** 2
    lam = np.sort(np.linalg.eigvalsh(L))[:7]
    print("Example 1: Laplacian on the circle, smallest eigenvalues:", np.round(lam, 3))
    print("           exact k^2:                                  [0 1 1 4 4 9 9]")
    print("           Dirac-type operator -i d/dtheta has eigenvalues k = 0, +-1, +-2, ... ;"
          " its square is the Laplacian")
    return lam


def example2_weyl(lams=(10, 100, 1000, 10000)):
    print("Example 2: Weyl's law on the circle")
    for lam in lams:
        k_max = int(np.floor(np.sqrt(lam)))
        N = 2 * k_max + 1                      # k = -k_max, ..., k_max
        print(f"           lambda = {lam:6d}: N = {N:4d}, 2 sqrt(lambda) = {2 * np.sqrt(lam):7.1f}")


def example3_two_points(m=0.5, grid=2001):
    """A = C^2 (functions (a, b) on two points), D = [[0, m], [m, 0]]."""
    D = np.array([[0, m], [m, 0]], dtype=float)
    best = 0.0
    for diff in np.linspace(-10, 10, grid):    # only a - b matters
        f = np.diag([diff, 0.0])
        if np.linalg.norm(D @ f - f @ D, 2) <= 1 + 1e-12:
            best = max(best, abs(diff))
    print(f"Example 3: two points, m = {m}: Connes distance = {best:.3f}, 1/|m| = {1 / abs(m):.3f}")
    return best


def example4_truncation(N=3):
    """Fourier modes -N..N. Multiplication by e^{i theta} shifts mode n to n + 1; keep only the window."""
    size = 2 * N + 1
    S = np.eye(size, k=-1)                     # truncated multiplication by e^{i theta}
    Sstar = S.T                                # truncated multiplication by e^{-i theta}
    C = S @ Sstar - Sstar @ S
    print(f"Example 4: truncation to modes -{N}..{N}")
    print("           as functions: e^{i theta} e^{-i theta} = e^{-i theta} e^{i theta} = 1")
    print("           truncated commutator diagonal:", np.diag(C).astype(int),
          " (nonzero only at the two edges of the window)")
    return C


if __name__ == "__main__":
    example1_circle()
    example2_weyl()
    example3_two_points()
    example4_truncation()
