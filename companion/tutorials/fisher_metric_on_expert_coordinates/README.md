# Rank of the Fisher metric on expert coordinates (GI theory side story)

Code for the Qiita article on whether Gluzman's Strategic Manifold, which has no metric, can be embedded into the business-environment manifold of GI theory (in Japanese).

On fictional data, the script runs four experiments:

- `toy`: a hand-computable example (2 expert coordinates to 3 indicators) for the Jacobian and the pullback metric
- `rank`: the Fisher metric has rank 1 when the only outcome is a single binary one
- `two_rulers`: compare the pullback metric and the Fisher metric at the same point and find the direction where they disagree
- `sigma_test`: test the functional form of the stability index Σ by comparing nested models (likelihood-ratio test)

## How to run

    pip install -r code/requirements.txt
    python code/gi_expert_rulers_jax.py             # all experiments
    python code/gi_expert_rulers_jax.py two_rulers  # a single experiment

All numbers are fictional. This is teaching material for illustrating the ideas.

## Author

Étale Cohomology

## License

License: Creative Commons Attribution 4.0 International (CC BY 4.0)
