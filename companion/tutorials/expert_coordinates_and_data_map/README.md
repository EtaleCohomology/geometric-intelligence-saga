# Connecting expert coordinates and the data map (GI theory side story)

Code for the Qiita article "Connecting expert coordinates and the data map" (in Japanese).

Two methods are checked on fictional data.

| Experiment | Content |
|---|---|
| `method2` | Method 2: build the map (VAE) first, then attach expert coordinates. Pullback metric on expert coordinates, minimum singular value, variation not explained by the expert coordinates, extrapolation check |
| `method1` | Method 1: learn expert coordinates as part of the latent space from the start. Prediction accuracy, leakage check, metric depending on z_free, and the metric near the cube's corners measured in two charts (expert coordinates and pre-sigmoid values) |
| `geodesic` | Draw a geodesic on the expert coordinates of the Method 2 map and compare it with the straight path |
| `diagnose` | Tell whether a collapsing direction is a model failure or a property of the data |
| `adversary` | A remedy for leakage (compete with a predictor that tries to recover expert coordinates from z_free) |
| `noise` | How much variability in expert ratings moves the metric and the geodesic length |
| `plus` | Method 2+: keep a per-case residual to obtain a per-case ruler without retraining the map; comparison with the true data movement |
| `fewlabels` | Method 2 with a linear map versus a network map, using 20, 50 and 200 labeled cases |
| `verbal` | Use verbal levels (low, medium, high) of expert coordinates as interval labels |
| `curvature` | Gaussian curvature on a 2D slice of the expert coordinates, and the stability of its sign |
| `gluzman` | Compare Gluzman's three-phase path with a geodesic under the data metric (fictional numbers; no conclusion about real strategy) |

## How to run

    pip install -r code/requirements.txt
    python code/gi_expert_coordinates_jax.py method2

All numbers are fictional. This is teaching material for illustrating the ideas.

## Author

Étale Cohomology

## License

License: Creative Commons Attribution 4.0 International (CC BY 4.0)
