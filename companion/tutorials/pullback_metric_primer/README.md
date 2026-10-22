# Pullback Metric Primer

An introductory tutorial on the Riemannian metric (pullback metric) in the latent space of Variational Autoencoders (VAEs).

## Target Audience

- Data scientists with limited mathematical background
- Corporate training programs and study groups
- Readers interested in the mathematical foundations of Geometric Intelligence (GI) theory

## Contents

The latent space produced by a VAE decoder appears to be a flat vector space, but it is in fact a curved space (a manifold) that stretches and contracts under the decoder mapping. This tutorial explains the concept of the **pullback metric** — the correct way to measure distances in this warped space — through three complementary lenses: equations, visualizations, and Python implementations.

Main topics:

- How a VAE compresses 100-dimensional data into a 2-dimensional strategic map
- The limits of Euclidean distance ("straight lines on the map are not the shortest path")
- Derivation of the Jacobian J_f and the metric tensor g = J_f^T J_f
- Mathematical background: what is the pullback metric?
- A PyTorch implementation of the metric
- Euclidean distance vs. Riemannian distance vs. information-geometric distance

## Document

📄 [pullback_metric_primer_JA.pdf](./pullback_metric_primer_JA.pdf) (Japanese)

## Prerequisites

- High school mathematics (partial derivatives, basic matrix operations)
- Basic Python (enough to read PyTorch code)
- Elementary linear algebra (vectors, inner products)

## Related Material

- [gi_theory_introduction_JA.pdf](../gi_theory_introduction_JA.pdf) — A systematic introduction to GI theory (48 pages, 6 chapters)

## License

CC BY 4.0
