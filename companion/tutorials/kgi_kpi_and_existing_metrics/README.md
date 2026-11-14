[English](./README.md) | [日本語](./README.ja.md)

# Speedometer and terrain: reading business intelligence from the map alongside existing metrics (Invitation to Geometric Intelligence, Part 5)

Code for Part 5 of the Qiita series "Invitation to Geometric Intelligence" (in Japanese).

On fictional data, existing metrics (VaR, expected shortfall, KRI) are read side by side with business intelligence obtained from the map (distance to the cliff). The map is a differentiable manifold built from socioeconomic data.

| Experiment | Content |
|---|---|
| `compare` | A phase where VaR is calm but the state approaches a cliff, and a phase where VaR jumps but the state stays at the bottom of a valley. The band obtained by re-estimating the map |
| `signals` | Checking "earlier warning" (hits, misses, false alarms, lead time, noise-to-signal ratio, evaluated on held-out periods) |
| `onepage` | The numbers for a one-page board view of the fictional company "Étale Heavy Industries" |

The article's figures (terrain with the cliff region, VaR vs distance to the cliff, hits/misses/false alarms) can be drawn from the same fictional data with `code/make_figures.py` (saved to `figs/`; a Japanese font is required).

## How to run

    pip install -r code/requirements.txt
    python code/gi_kgi_kpi_jax.py compare
    cd code && python make_figures.py

All numbers are fictional. Étale Heavy Industries is a fictional company. This is teaching material for illustrating the ideas.

## Author

Étale Cohomology

## License

License: Creative Commons Attribution 4.0 International (CC BY 4.0)
