# Giving meaning to a VAE's latent space with labels

Code for the Qiita article "Giving meaning to a VAE's latent space with labels: four representative papers on semi-supervised and conditional VAEs" (in Japanese).

On fictional data, the script runs the following experiments:

- `labels`: vary the number of labeled examples (0, 20, 200) and compare informativeness (R² of each factor from the whole latent space) with alignment (R² of each factor from its own axis)
- `compare`: with the same 20 labels, compare joint training with post-hoc conditioning (a linear semantic direction), over 5 random seeds
- `manipulate`: move along an axis or a semantic direction and check, with an oracle reader of the true factors, whether only the targeted factor changes
- `correlated`: train on data with correlated factors and evaluate on data with independent factors

## How to run

    pip install -r code/requirements.txt
    python code/gi_few_labels_jax.py            # all experiments (a few minutes on CPU)
    python code/gi_few_labels_jax.py compare    # a single experiment

All numbers are fictional. This is teaching material for illustrating the ideas.

## Author

Étale Cohomology

## License

License: Creative Commons Attribution 4.0 International (CC BY 4.0)
