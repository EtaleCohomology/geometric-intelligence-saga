# Geometric Intelligence — Theory

> 🇯🇵 [日本語版はこちら / Japanese version](README.ja.md)

This directory contains the theoretical treatises on **Geometric Intelligence (GI)**,
a differential-geometric framework for decision-making and reinforcement learning
on data-driven Riemannian manifolds.

## Overview

Standard analytical methods in business and policy — regression, PCA, linear optimisation —
implicitly assume that the underlying state space is Euclidean. Real-world environments,
however, are intrinsically nonlinear: the same action produces qualitatively different
outcomes depending on where in the state space it is taken.

Geometric Intelligence theory addresses this gap by:

1. **Constructing** the management or policy environment as a data-driven Riemannian
   manifold via the pullback metric of a VAE decoder,
   $g_{ij}(z) = (J_{f_\theta}^\top J_{f_\theta})_{ij}$.
2. **Analysing** this manifold with the full differential-geometric toolkit —
   covariant derivatives, curvature tensors, Lie derivatives, geodesics, and
   optimal control via Pontryagin's maximum principle.
3. **Extending** the framework to deep reinforcement learning by identifying and
   correcting four geometric errors that all existing World Models architectures
   commit on curved state spaces.

## Repository structure

```
theory/
├── README.md              # This file (English)
├── README.ja.md           # Japanese version
├── Vol1/                  # Volume 1: Mathematical foundations and the GI pipeline
│   ├── GI_JP_Theory_Comprehensive_Paper.pdf      # Japanese, comprehensive
│   ├── GI_Theory_Full_Version_EN.pdf             # English, full version
│   └── GI_EN_Theory_NeurIPS_Format.pdf           # English, NeurIPS-format short version
└── Vol2/                  # Volume 2: Riemannian World Models for DRL
    ├── GI_JP_Theory_Comprehensive_Paper_Vol2.pdf # Japanese
    └── GI_EN_Theory_NeurIPS_Format_Vol2.pdf      # English, NeurIPS-format
```

All papers within a volume present the same content; choose the language and length
that best suit your needs.

## Volume 1 — Geometric Intelligence: A Differential-Geometric Decision-Making Framework on Data-Driven Differentiable Manifolds

Volume 1 establishes the mathematical foundations of GI theory and presents the
end-to-end pipeline from raw data to actionable intelligence. The central thesis
is that business and policy environments are *curved, not flat*, and that
neglecting this curvature induces the same category of structural error as
navigating the Earth's surface by means of a flat map.

**Key results:**

- **Proposition 2.1 (foundational).** A VAE decoder satisfying four conditions —
  domain compactness, smoothness ($C^k$ with $k \geq 3$), full-rank Jacobian,
  and injectivity — yields an embedded submanifold equipped with a well-defined
  Riemannian metric, *independent of industry or data distribution*.
- **A 10-step pipeline** from data collection through manifold construction,
  curvature computation, Lie derivative analysis, geodesic computation, and
  optimal control simulation via Pontryagin's maximum principle, to visualisation
  and decision-making.
- **Five pillars of reliability** for trustworthy outputs: MC Dropout (uncertainty
  quantification), SHAP (explainability), Double Machine Learning (causal
  inference), ZKML (cryptographic verification), and formal verification
  (Coq, Agda, Lean 4, Z3).
- **Seven geometric extensions** systematising the theoretical reach: Morse
  theory (tipping-point classification), exotic manifolds with optimal transport
  (early detection of structural shifts), surgery theory (radical restructuring),
  information geometry, symplectic geometry (long-horizon stability), Weyl
  geometry (currency-gauge invariance), and Sheaf theory (local consistency).
- **Two case studies** demonstrating the full pipeline: a mid-cap manufacturing
  firm (6-dimensional management-environment manifold) and a national security
  council (8-dimensional international-affairs manifold across 193 countries
  and 30 years of data).

**Keywords.** Riemannian manifold, pullback metric, variational autoencoder, Lie
derivative, covariant derivative, curvature tensor, geodesic, optimal control,
Christoffel symbols, data-driven manifold, deep reinforcement learning, digital
twin, formal verification, MC Dropout, SHAP, DML, ZKML.

## Volume 2 — Riemannian World Models: Deep Reinforcement Learning on Data-Driven Differentiable Manifolds

Volume 2 extends the GI framework to deep reinforcement learning. The central
contribution is the identification and formal separation of two distinct
Riemannian manifolds that coexist in every DRL system, and the demonstration
that all prior geometric DRL research has addressed only the first while
systematically ignoring the second.

**Key results:**

- **The two-manifold structure of DRL (the foundational observation).**
  - The first manifold $\mathcal{M}_{\text{weights}}$ — the policy parameter space
    with the Fisher information metric — governs *how an agent learns*. The
    Natural Policy Gradient (Kakade, 2001) and its descendants TRPO and PPO
    correctly handle its geometry.
  - The second manifold $\mathcal{M}_{\text{env}}$ — the environmental state space
    with the pullback metric — governs *where the agent acts*. **Every existing
    World Models architecture (DreamerV3, DreamerV2, PlaNet, …) treats it as
    Euclidean, which is geometrically incorrect on curved state spaces.**
- **Four geometric errors** of Euclidean World Models on curved
  $\mathcal{M}_{\text{env}}$: Euclidean state transition (the agent walks off
  the manifold), Euclidean policy gradient (wrong direction of steepest ascent),
  trivial parallel transport (incomparable tangent vectors), and Euclidean
  distance reward (wrong intrinsic distance).
- **Riemannian World Models (RWM)** — four geometrically correct replacements:
  exponential-map transition $z_{t+1} = \exp_{z_t}(f_\theta(z_t, a_t))$,
  Riemannian gradient $g^{ij}\partial_j V$, Levi-Civita parallel transport for
  temporal credit assignment, and geodesic distance reward $r = -d_g(z_{t+1}, z^*)^2$.
  All four corrections come with PyTorch-compatible implementations via
  automatic differentiation, together with a geometric consistency theorem
  (Theorem 4.3) establishing manifold preservation, coordinate independence,
  flat-space recovery of DreamerV3, and steepest-ascent uniqueness.
- **Four-layer gap analysis** explaining why this structural separation has gone
  unrecognised: data sparsity and educational gaps in the social sciences; the
  known-manifold assumption in engineering and physics; the two-manifold
  confusion in machine learning; and the disciplinary intersection problem
  requiring simultaneous expertise in differential geometry, VAEs, deep RL,
  and domain-specific data.
- **Multi-agent extension and a Finsler roadmap** — for $N$ heterogeneous
  agents, $\mathcal{M}_{\text{env}}$ is shared and inter-agent messages must
  be parallel-transported. The framework extends naturally to Finsler manifolds
  (required for aviation routing under wind, wind energy, maritime logistics,
  and supply chains, where the metric is directionally asymmetric) and to
  time-varying Finsler manifolds (where the metric depends on time, as with
  seasonal jet streams), both currently open problems.

**Keywords.** Two-manifold structure, Riemannian World Models, exponential map,
Riemannian gradient, Levi-Civita parallel transport, geodesic distance reward,
DreamerV3, World Models, deep reinforcement learning, Fisher information metric,
pullback metric, Finsler manifold.

## How to read

- **For a quick overview in English:** Vol.1 NeurIPS-format short version.
- **For the full mathematical development in English:** Vol.1 full version, then
  Vol.2.
- **For Japanese readers:** Vol.1 comprehensive paper (JP), then Vol.2 (JP).

Vol.1 is the prerequisite for Vol.2: the data-driven Riemannian manifold
$\mathcal{M}_{\text{env}}$ constructed in Vol.1 is the substrate on which the
DRL agents of Vol.2 learn and act.

## Citation

```bibtex
@misc{etale2026gi_vol1,
  author = {Etale Cohomology},
  title  = {Geometric Intelligence, Volume 1: A Differential-Geometric
            Decision-Making Framework on Data-Driven Differentiable Manifolds},
  year   = {2026}
}

@misc{etale2026gi_vol2,
  author = {Etale Cohomology},
  title  = {Geometric Intelligence, Volume 2: Riemannian World Models —
            Deep Reinforcement Learning on Data-Driven Differentiable Manifolds},
  year   = {2026}
}
```

## License

This work is licensed under a
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

You are free to share and adapt the material for any purpose, including commercially,
provided that appropriate credit is given.

