[English](./README.md) | [日本語](./README.ja.md)

# Reading the Shape of "Where We Stand Now" on a Socioeconomic Map

An introductory tutorial that explains, through a map analogy, how to work on a map (a VAE latent space) that has passed the three conditions: measuring true distances, finding "cliffs" by the sign of curvature, finding valleys, passes and peaks with Morse theory, and reading the map's connectivity and inconsistencies with persistent homology and sheaves. This folder contains the code for the second article in the Qiita series "Geometric Intelligence".

## Target Audience

- Python machine learning engineers working in companies and government agencies
- Management consultants
- Policy consultants
- Policy staff and secretaries of members of parliament who are familiar with machine learning (AI) and AI agents

## Contents

- "True distance" measured with the Riemannian metric
- Scalar curvature and the meaning of its sign
- Morse theory: critical points (valleys, passes, peaks), degenerate critical points, the water-level analogy, and tipping points when a valley becomes shallow
- Persistent homology and sheaves, and their relation to network analysis
- Computations on curved spaces in physics and engineering
- Python code (JAX) that checks everything from the three conditions to curvature and critical points on shapes with known answers (a sphere and a saddle)

## Document

📄 [reading_current_position_shape_JA.pdf](./reading_current_position_shape_JA.pdf) (Japanese). A PDF version of the Qiita article published on November 5, 2026: https://qiita.com/etale_cohomology/items/940ecb65278977f3d7fd

## Code

- [code/gi_manifold_check_curvature_morse_jax.py](./code/gi_manifold_check_curvature_morse_jax.py)

```
pip install -r code/requirements.txt
python code/gi_manifold_check_curvature_morse_jax.py sphere 0.001
python code/gi_manifold_check_curvature_morse_jax.py sphere 0.05
python code/gi_manifold_check_curvature_morse_jax.py saddle 0.05
python code/gi_manifold_check_curvature_morse_jax.py saddle 0.05 3   # optional 3rd argument: training random seed (default 1)
```

The results table and the positions of the valleys and the pass in the article were obtained by running this code in the author's environment.

## Suggested Reading Order

1. [Part 1](../vae_to_socioeconomic_map/) — checking the map (may a ruler be laid on it?)
2. Part 2 (this tutorial) — reading the shape of where we stand now
3. [gi_theory_introduction_JA.pdf](../gi_theory_introduction_JA.pdf) — a systematic introduction to GI theory

## Related Material

- The fictional settings discussed in the article (case study novels by industry and policy domain): [case-studies](../../../case-studies/)

## Note

The case study content introduced in the article consists of fictional story settings written by the author. It does not describe findings obtained from real data of real companies or government agencies.

## License

CC BY 4.0
