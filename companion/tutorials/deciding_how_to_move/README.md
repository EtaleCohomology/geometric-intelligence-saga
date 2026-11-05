[English](./README.md) | [日本語](./README.ja.md)

# Deciding "How to Move" on a Socioeconomic Map

An introductory tutorial that explains, through a map analogy, how to work on a map (a VAE latent space) that has passed the three conditions: choosing the most natural path (geodesic), measuring the effect of the order of policies (Lie brackets), treating policies as interventions in a flow (Neural ODE, optimal control), and checking how reliable a path is. This folder contains the code for the third article in the Qiita series "Geometric Intelligence".

## Target Audience

- Python machine learning engineers working in companies and government agencies
- Management consultants
- Policy consultants
- Policy staff and secretaries of members of parliament who are familiar with machine learning (AI) and AI agents
- Researchers and research scientists interested in applying advanced mathematics as practical tools for business and policy analysis

## Code

| File | Content |
|---|---|
| [code/geodesic_hill_jax.py](./code/geodesic_hill_jax.py) | Experiment 1: checking that the geodesic goes around a hill |
| [code/lie_bracket_parking_jax.py](./code/lie_bracket_parking_jax.py) | Experiment 2: checking the Lie bracket with parallel parking |
| [code/neural_ode_flow_jax.py](./code/neural_ode_flow_jax.py) | Experiment 3: relearning a flow and the skeleton of the terrain (valleys and a pass) with a Neural ODE |

```
pip install -r code/requirements.txt
python code/geodesic_hill_jax.py 0.05      # 1st argument: KL weight β
python code/geodesic_hill_jax.py 0.05 3    # optional 2nd argument: training random seed (default 1)
python code/lie_bracket_parking_jax.py
python code/neural_ode_flow_jax.py
```

The numbers in the article were obtained by running this code in the author's environment.

## Suggested Reading Order

1. [Part 1](../vae_to_socioeconomic_map/) — checking the map (may a ruler be laid on it?)
2. [Part 2](../reading_current_position_shape/) — reading the shape of where we stand now
3. Part 3 (this tutorial) — deciding how to move
4. [gi_theory_introduction_JA.pdf](../gi_theory_introduction_JA.pdf) — a systematic introduction to GI theory

## Related Material

- The fictional settings discussed in the article (case study novels by industry and policy domain): [case-studies](../../../case-studies/)

## Note

The case study content introduced in the article consists of fictional story settings written by the author. It does not describe findings obtained from real data of real companies or government agencies.

## License

CC BY 4.0
