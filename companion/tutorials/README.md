[English](./README.md) | [日本語](./README.ja.md)

# Tutorials

Educational material introducing Geometric Intelligence (GI) theory to new readers.

## Purpose

The tutorials in this directory are entry-level resources designed to bridge the gap between general familiarity with data science and the mathematical machinery used in Volumes 1 and 2 of GI theory. They assume high-school mathematics and build up the necessary differential-geometric vocabulary (Riemannian metrics, Lie derivatives, geodesics, optimal control) through analogies, worked examples, and Python implementation snippets.

These tutorials are not substitutes for Volumes 1 and 2. They prepare readers to engage with the main theory, and they serve as reference material for readers who want to recall a concept quickly.

## Contents

For first-time readers, we recommend the following order: 
`vae_to_socioeconomic_map/` → `gi_theory_introduction_JA.pdf` → `gi_metrics_and_manifolds_overview_JA.pdf` → `gi_practice_materials/`

### `vae_to_socioeconomic_map/`

*Turning a VAE latent space into a "map for socioeconomic analysis": three conditions for introducing a Riemannian metric.*

The first article in the Qiita series "Geometric Intelligence" (first published March 10, 2026; fully revised November 1, 2026). It covers:

- How a VAE works and why its latent space is built from probability distributions (including β-VAE and disentanglement)
- Why the latent space of socioeconomic data cannot be used as a "map" as it is
- The three conditions for laying a ruler (Riemannian metric) on a latent space, plus smoothness
- Data constraints by industry and policy domain, and the trial and error needed to satisfy the conditions (from fictional case studies)

Japanese only (for now). Approximately 27 pages. Recommended as the first document to read in this directory.

### `gi_theory_introduction_JA.pdf`

*An introduction to GI theory for readers with high-school mathematics.*

A self-contained introduction covering:

- Chapter 1: VAE, pullback metric, and the construction of differentiable manifolds
- Chapter 2: Ten concrete insights that GI theory provides to executives and policy-makers
- Chapter 3: How physics, chemistry, and engineering have used this mathematics for a century
- Chapter 4: Why this approach has not previously been available for socio-economic data
- Chapter 5: Alternative mathematical frameworks (information geometry, optimal transport, diffusion geometry, neural ODE, GPLVM)
- Chapter 6: Python implementation examples using PyTorch, torchdiffeq, pydiffmap, POT, and related libraries

Japanese only (for now). Approximately 48 pages.

### `gi_metrics_and_manifolds_overview_JA.pdf`

*An overview of the metric and manifold concepts most likely to be relevant to industrial applications of GI theory.*

A parallel-track training document: a "basic" track for readers with high-school mathematics and an "advanced" track for undergraduate-level readers, presented side by side. It covers:

- Chapter 1: Manifolds and maps (differentiable structure, the level of smoothness required by GI theory, tangent spaces)
- Chapter 2: Riemannian metrics and their generalisations (pullback metric, positive definiteness and the full-rank condition, connections and curvature)
- Chapter 3: Finsler and Randers metrics (distance in a space with "wind", Zermelo's navigation problem)
- Chapter 4: H¹ cohomology (holes and paths, zigzag persistence)
- Chapter 5: The mathematics of geodesics
- Chapter 6: Optimal control (Hamiltonians, adjoint variables, Pontryagin's maximum principle)
- Chapter 7: Other geometries used in the series (symplectic, Weyl, information geometry, hyperbolic, sub-Riemannian, optimal transport)

Japanese only (for now). Approximately 31 pages.

### `gi_practice_materials/`

*GI theory practice materials (four documents).*

Teaching materials that use the case study novels (by industry and policy domain) to organise, across the works, the workflow from building a manifold from data (the quality checklist), to analysing it, to demonstrating to stakeholders that the pipeline is correct. The folder contains four documents: the quality checklist (standard and detailed editions), analyses on the finished manifold, and verification and accountability.

Japanese only (for now). See the README in the folder for details of each document.

## Licence

CC BY 4.0.

## Contact

etalecohomology2026@proton.me
