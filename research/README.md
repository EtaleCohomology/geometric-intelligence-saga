[English](./README.md) | [日本語](./README.ja.md)

# research/

Research notes, mathematical explorations, and implementation studies toward Volume 3 of Geometric Intelligence (GI) theory.

## Scope

This directory contains work in progress. The material here is not part of the established GI theory presented in Volumes 1 and 2, nor of the applied papers in `papers/`. It documents ongoing investigations into extensions of the theory: alternative mathematical frameworks, implementation trade-offs, and surveys of related literature.

The current GI theory uses a variational autoencoder (VAE) trained on observed data, with the decoder defining a pullback metric on the latent space. This approach is mathematically valid and has produced useful results across multiple case studies. Alternative frameworks — neural ODEs, optimal transport, information geometry, diffusion geometry, Gaussian process latent variable models — offer different trade-offs along several axes: mathematical rigor, scalability, uncertainty quantification, and suitability for control simulation.

The goal of this directory is to evaluate these alternatives systematically, prototype them in Python where useful, and identify which extensions are most promising for Volume 3.

## Subdirectories

### `mathematical_foundations/`

Comparative analysis of mathematical frameworks for GI theory. Includes evaluations of alternatives to the VAE + pullback metric approach, with attention to mathematical rigor, empirical fit with socio-economic data, computational cost, and the kind of insights each framework provides to decision-makers.

### `python_implementations/`

Prototype implementations and benchmarks. Code studies of specific mathematical constructs (Lie derivatives via automatic differentiation, geodesic distance computation, condition number monitoring, neural ODE integration on manifolds).

### `literature_reviews/`

Surveys of related work. Systematic comparisons with existing approaches in geometric deep learning, information geometry, optimal transport, manifold learning, and control theory on manifolds.

## Style

The working language of this directory is English in the style used by AI research scientists at institutions such as MIT, OpenAI, and Anthropic: direct, active voice, precise technical vocabulary, empirically grounded. This differs from the Oxbridge prose of the applied papers in `papers/` — the difference is deliberate and reflects the distinct character of the work (technical exploration versus philosophical essay).

The Japanese version of each document follows the style of contemporary AI research writing (Amari Shun'ichi, Matsuo Yutaka), which is already compatible with the MIT-style English.

## Status

This directory is under active development. Files may be added, revised, or removed without notice. Substantive contributions will be documented in commit messages and, where appropriate, given dedicated files.

## Licence

CC BY 4.0, consistent with the rest of the repository.

## Contact

etalecohomology2026@proton.me
