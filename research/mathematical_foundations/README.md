[English](./README.md) | [日本語](./README.ja.md)

# mathematical_foundations/

Comparative analysis of mathematical frameworks for Geometric Intelligence (GI) theory.

## Purpose

Volumes 1 and 2 of GI theory build on a single mathematical foundation: a variational autoencoder (VAE) trained on observed data, with the decoder defining a pullback metric on the latent space. This foundation is sound but not unique. Several alternative frameworks can, in principle, support the same geometric analyses — Lie derivatives, scalar curvature, geodesics, control on manifolds.

This subdirectory compares these alternatives along multiple axes:

- Mathematical rigor (guarantees on smoothness, non-degeneracy, diffeomorphism)
- Scalability to high-dimensional data
- Suitability for socio-economic and humanities data
- Computational cost
- Quality of insights delivered to decision-makers
- Compatibility with control simulation

## Frameworks Under Review

The main alternatives currently tracked:

- **Neural ODE / Normalizing flows** — diffeomorphism guaranteed by construction. Natural fit with control theory.
- **Optimal transport** — Wasserstein distance between distributions. Strong mathematical foundations.
- **Information geometry** — Fisher information metric on statistical manifolds. Tight integration with statistical inference.
- **Diffusion geometry** — spectral methods on graphs. Robust to noise.
- **Gaussian process latent variable models (GPLVM)** — smooth by construction. Built-in uncertainty quantification.
- **Other** — hyperbolic embeddings, symplectic geometry, Grassmann manifolds, Weyl geometry (as relevant).

## Files

Files will be added as the analysis progresses. Each file addresses a specific framework or comparison question.

## Method

Each framework is evaluated on identical criteria. Where a framework is particularly strong in one dimension (e.g., neural ODE for control, GPLVM for uncertainty), the report documents both the strength and the trade-off required to access it.

The goal is not to replace the current VAE-based foundation but to identify which extensions warrant integration into Volume 3.
