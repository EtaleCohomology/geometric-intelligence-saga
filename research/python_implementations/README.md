[English](./README.md) | [日本語](./README.ja.md)

# python_implementations/

Prototype implementations and benchmarks for Geometric Intelligence (GI) theory.

## Purpose

This subdirectory contains working code. The material here is exploratory: implementations of specific mathematical constructs, benchmarks comparing approaches, and small experiments that inform the theoretical analysis in `mathematical_foundations/`.

Code here is not production-ready. It is instrumented for understanding, not for deployment. When a prototype matures into a reusable tool, it migrates to `companion/`.

## Scope

Typical content:

- Lie derivative computation via PyTorch autograd (`create_graph=True`)
- Pullback metric construction from a trained VAE decoder
- Condition number monitoring during metric computation
- Christoffel symbols, scalar curvature, geodesic integration
- Neural ODE with `torchdiffeq` for manifold dynamics
- Comparison benchmarks between frameworks on synthetic data

## Files

Files will be added as prototypes develop. Expect both Python scripts (`.py`) and Jupyter notebooks (`.ipynb`).

## Conventions

- Python 3.10+
- PyTorch for automatic differentiation
- `torchdiffeq` for neural ODE
- NumPy and SciPy for classical numerical work
- Matplotlib for visualization
- Random seeds fixed where reproducibility matters
- Each file opens with a docstring stating purpose, inputs, outputs, and limitations

## Dependencies

A `requirements.txt` will be added when the first runnable script is committed.

## Licence

CC BY 4.0 for documents; implementations may carry MIT or equivalent permissive licences. Each file states its own licence in the header.
