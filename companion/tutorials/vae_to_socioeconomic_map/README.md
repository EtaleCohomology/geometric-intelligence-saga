[English](./README.md) | [日本語](./README.ja.md)

# Turning a VAE Latent Space into a "Map for Socioeconomic Analysis"

An introductory tutorial that explains, through a map analogy, the three conditions that must be checked before a Riemannian metric — a "ruler" — can be applied to the latent space of a Variational Autoencoder (VAE). This is a PDF version of the first article in the Qiita series "Geometric Intelligence" (first published March 10, 2026; fully revised November 1, 2026).

## Target Audience

- Python machine learning engineers working in companies and government agencies
- Management consultants
- Policy consultants
- Policy staff and secretaries of members of parliament who are familiar with machine learning (AI) and AI agents

## Contents

The latent space of a VAE performs very well for image generation. However, when the latent space of socioeconomic data is used as a "map" to measure distances and curvature, it does not give correct answers as it is: the map may be torn (split into islands), folded (distant places overlap), or squashed (the scale becomes zero in some places). This tutorial explains, with almost no equations, the conditions under which a latent space becomes "a map on which a ruler may be laid".

Main topics:

- How a VAE works (encoder, sampling, decoder) and why its latent space is built from probability distributions
- β-VAE and disentanglement (column: VAE and GAN)
- Problems that arise when the latent space of socioeconomic data is used as a "map"
- The three conditions for applying a ruler (Riemannian metric) to the map, plus smoothness (with practical thresholds for engineers)
- An intuitive explanation of "manifolds" and "differentiable manifolds"
- Data constraints by industry and policy domain, and the trial and error needed to satisfy the conditions (from fictional case studies)

## Document

📄 [vae_to_socioeconomic_map_JA.pdf](./vae_to_socioeconomic_map_JA.pdf) (Japanese)

## Prerequisites

- Basic machine learning vocabulary (training, model, latent space, etc.)
- The mathematics is written to be readable with high school mathematics. Only the practical thresholds for engineers use the terms Jacobian matrix and singular value

## Suggested Reading Order

1. This tutorial (series part 1) — the conditions for laying a ruler on a latent space ([Qiita article](https://qiita.com/etale_cohomology/items/4797140237582e711c88))
2. [Series part 2](../reading_current_position_shape/) — reading the shape of "where we stand now" on a map that satisfies the conditions: true distances with the Riemannian metric, "cliffs" by the sign of curvature, valleys, passes and peaks with Morse theory, and connectivity with persistent homology and sheaves ([Qiita article](https://qiita.com/etale_cohomology/items/940ecb65278977f3d7fd))
3. [gi_theory_introduction_JA.pdf](../gi_theory_introduction_JA.pdf) — a systematic introduction to GI theory

## Related Material

- The fictional settings discussed in this tutorial (case study novels by industry and policy domain): [case-studies](../../../case-studies/)

## Note

The data constraints and trial and error described in Section 5 are events in fictional case studies created by the author. They are not findings obtained from real data of real companies or government agencies.

## License

CC BY 4.0
