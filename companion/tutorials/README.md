[English](./README.md) | [日本語](./README.ja.md)

# Tutorials

Educational material introducing Geometric Intelligence (GI) theory to new readers.

## Purpose

The tutorials in this directory are entry-level resources designed to bridge the gap between general familiarity with data science and the mathematical machinery used in Volumes 1 and 2 of GI theory. They assume high-school mathematics and build up the necessary differential-geometric vocabulary (Riemannian metrics, Lie derivatives, geodesics, optimal control) through analogies, worked examples, and Python implementation snippets.

These tutorials are not substitutes for Volumes 1 and 2. They prepare readers to engage with the main theory, and they serve as reference material for readers who want to recall a concept quickly.

## Contents

For first-time readers, we recommend the following order: `vae_to_socioeconomic_map/` → `gi_theory_introduction_JA.pdf` → `gi_metrics_and_manifolds_overview_JA.pdf`

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
- Chapter 2: Riemannian metrics
