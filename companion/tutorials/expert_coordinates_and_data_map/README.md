# Connecting expert coordinates and the data map (GI theory side story)

Code for the Qiita article "Connecting expert coordinates and the data map" (in Japanese).

Two methods are checked on fictional data.

| Experiment | Content |
|---|---|
| `method2` | Method 2: build the map (VAE) first, then attach expert coordinates. Pullback metric on expert coordinates, minimum singular value, variation not explained by the expert coordinates, extrapolation check |
| `method1` | Method 1: learn expert coordinates as part of the latent space from the start. Prediction accuracy, leakage check, metric depending on z_free, metric collapse near the cube's corners |
| `geodesic` | Draw a geodesic on the expert coordinates of the Method 2 map and compare it with the straight path |
| `diagnose` | Tell whether a collapsing direction is a model failure or a property of the data |
| `adversary` | A remedy for leakage (compete with a predictor that tries to recover expert coordinates from z_free) |
| `noise` | How much variability in expert ratings moves the metric and the geodesic length |

## How to run

    pip install -r code/requirements.txt
    python code/gi_expert_coordinates_jax.py method2

All numbers are fictional. This is teaching material for illustrating the ideas.

## Author

Étale Cohomology

## License

License: Creative Commons Attribution 4.0 International (CC BY 4.0)
