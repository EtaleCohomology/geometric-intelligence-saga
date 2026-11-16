[English](./README.md) | [日本語](./README.ja.md)

# Tutorials

Educational material introducing Geometric Intelligence (GI) theory to new readers.

## Purpose

The tutorials in this directory are entry-level resources designed to bridge the gap between general familiarity with data science and the mathematical machinery used in Volumes 1 and 2 of GI theory. They assume high-school mathematics and build up the necessary differential-geometric vocabulary (Riemannian metrics, Lie derivatives, geodesics, optimal control) through analogies, worked examples, and Python implementation snippets.

These tutorials are not substitutes for Volumes 1 and 2. They prepare readers to engage with the main theory, and they serve as reference material for readers who want to recall a concept quickly.

## Contents

For first-time readers, we recommend the following order: 
`vae_to_socioeconomic_map/` → `reading_current_position_shape/` → `gi_theory_introduction_JA.pdf` → `gi_metrics_and_manifolds_overview_JA.pdf` → `gi_practice_materials/` → `deciding_how_to_move/` → `building_map_from_real_data/` → `kgi_kpi_and_existing_metrics/` → `labels_give_meaning_to_vae/` → `expert_coordinates_and_data_map/` → `fisher_metric_on_expert_coordinates/`

### `vae_to_socioeconomic_map/`

*Turning a VAE latent space into a "map for socioeconomic analysis": three conditions for introducing a Riemannian metric.*

The first article in the Qiita series "Geometric Intelligence" (first published March 10, 2026; fully revised November 1, 2026). It covers:

- How a VAE works and why its latent space is built from probability distributions (including β-VAE and disentanglement)
- Why the latent space of socioeconomic data cannot be used as a "map" as it is
- The three conditions for laying a ruler (Riemannian metric) on a latent space, plus smoothness
- Data constraints by industry and policy domain, and the trial and error needed to satisfy the conditions (from fictional case studies)

Japanese only (for now). Approximately 27 pages. Recommended as the first document to read in this directory.

### `reading_current_position_shape/`

*Reading the shape of "where we stand now" on a socioeconomic map: finding "cliffs" and "passes" with curvature and Morse theory.*

The second article in the Qiita series "Geometric Intelligence" (published November 5, 2026). It covers:

- True distances measured with the Riemannian metric, and reading the sign of scalar curvature
- Morse theory: valleys, passes and peaks, degenerate critical points, and tipping points (including early-warning indicators of slowing recovery)
- Persistent homology, Mapper and sheaves, and how they differ from widely used methods (UMAP, clustering, anomaly detection, VaR)
- How TDA differs from GI theory
- Python code (JAX) that checks the three conditions, curvature and critical points on shapes with known answers

Japanese only (for now). Approximately 50 pages, with code. Recommended as the second document to read in this directory.
### `deciding_how_to_move/`

*Deciding "how to move" on a socioeconomic map.*

Code for the third article in the Qiita series "Geometric Intelligence". It covers:

- Choosing the most natural path (geodesic) on a map that has passed the three conditions
- The effect of the order of policies (Lie brackets)
- Treating policies as interventions in a flow (Neural ODE, optimal control), and checking how reliable a path is
- Python code (JAX) for three experiments: a geodesic going around a hill, the Lie bracket in parallel parking, and relearning a flow and the skeleton of the terrain with a Neural ODE

Japanese only (for now). With code.

### `building_map_from_real_data/`

*Building a "map" from real socioeconomic data.*

Code for the fourth article in the Qiita series "Geometric Intelligence". It covers:

- Choosing data backwards from the question the map should answer
- Real-data problems (record linkage, granularity, units, missing values, outliers, reporting lags and revisions, structural change, data that cannot leave an organisation) and how to treat them
- Diagnosing which of the three conditions failed, and how to fix it
- Python code (JAX) that turns a shape with a known answer into deliberately messy data and checks how different preprocessing changes the three conditions

Japanese only (for now). With code.

### `kgi_kpi_and_existing_metrics/`

*Speedometer and terrain: reading business intelligence from the map alongside existing metrics.*

Code for Part 5 of the Qiita series "Invitation to Geometric Intelligence". It covers:

- Reading existing metrics (VaR, expected shortfall, KRI) side by side with business intelligence from the map (distance to the cliff), on fictional data
- Checking "earlier warning" (hits, misses, false alarms, lead time; evaluated on held-out periods)
- The numbers for a one-page board view of the fictional company "Étale Heavy Industries"
- Code to draw the article's figures

Japanese only (for now). With code.

### `gi_theory_introduction_JA.pdf`

*An introduction to GI theory for readers with high-school mathematics.*

A self-contained introduction covering:

- Chapter 1: VAE, pullback metric, and the construction of differentiable manifolds
- Chapter 2: Ten concrete insights that GI theory provides to executives and policy-makers
- Chapter 3: How physics, chemistry, and engineering have used this mathematics for a century
- Chapter 4: Why this approach has not previously been available for socio-economic data
- Chapter 5: Alternative mathematical frameworks (information geometry, optimal transport, diffusion geometry, neural ODE, GPLVM)
- Chapter 6: Python implementation examples using PyTorch, torchdiffeq, pydiffmap, POT, and related libraries

Japanese only (for now). Approximately 48 pages.

### `gi_metrics_and_manifolds_overview_JA.pdf`

*An overview of the metric and manifold concepts most likely to be relevant to industrial applications of GI theory.*

A parallel-track training document: a "basic" track for readers with high-school mathematics and an "advanced" track for undergraduate-level readers, presented side by side. It covers:

- Chapter 1: Manifolds and maps (differentiable structure, the level of smoothness required by GI theory, tangent spaces)
- Chapter 2: Riemannian metrics and their generalisations (pullback metric, positive definiteness and the full-rank condition, connections and curvature)
- Chapter 3: Finsler and Randers metrics (distance in a space with "wind", Zermelo's navigation problem)
- Chapter 4: H¹ cohomology (holes and paths, zigzag persistence)
- Chapter 5: The mathematics of geodesics
- Chapter 6: Optimal control (Hamiltonians, adjoint variables, Pontryagin's maximum principle)
- Chapter 7: Other geometries used in the series (symplectic, Weyl, information geometry, hyperbolic, sub-Riemannian, optimal transport)

Japanese only (for now). Approximately 31 pages.

### `gi_practice_materials/`

*GI theory practice materials (four documents).*

Teaching materials that use the case study novels (by industry and policy domain) to organise, across the works, the workflow from building a manifold from data (the quality checklist), to analysing it, to demonstrating to stakeholders that the pipeline is correct. The folder contains four documents: the quality checklist (standard and detailed editions), analyses on the finished manifold, and verification and accountability.

Japanese only (for now). See the README in the folder for details of each document.

### `labels_give_meaning_to_vae/`

*Giving meaning to a VAE's latent space with labels: four representative papers on semi-supervised and conditional VAEs.*

Code for a Qiita article (to be published soon). It covers:

- Why the axes of a latent space learned without labels usually have no meaning (Locatello et al., 2019)
- Representative papers that give meaning to the axes with a few labels (semi-supervised VAE, latent constraints, conditional subspace VAE, VAE for regression), and semantic directions
- Comparing informativeness and alignment as the number of labels changes, and comparing joint training with post-hoc conditioning
- Python code (JAX) that checks these ideas on fictional data

Japanese only (for now). With code.

### `expert_coordinates_and_data_map/`

*GI theory side story: connecting expert coordinates and the data map — bringing Gluzman's Strategic Manifold into Geometric Intelligence.*

Code for a Qiita article (to be published soon). It covers:

- Two methods for connecting coordinates whose meaning is defined by human experts (the four axes of Gluzman's Strategic Manifold) to a data map built with a VAE
- The pullback metric (ruler) on expert coordinates, the minimum singular value, geodesics and curvature
- Why to start with a linear map when there are few cases, keeping a per-case residual (Method 2+), and verbal levels of expert coordinates
- Handling the corners of the cube (how the apparent metric depends on the chart used)
- Python code (JAX) that checks these ideas on fictional data

Japanese only (for now). With code.

### `fisher_metric_on_expert_coordinates/`

*GI theory side story: can Gluzman's Strategic Manifold, which has no ruler (metric), be embedded into the business-environment manifold of GI theory? — measuring human experts' coordinates with the Fisher metric.*

Code for a Qiita article (published November 16, 2026: https://qiita.com/etale_cohomology/items/7c083c3021b0b826c892 ). It covers:

- Two rulers on expert coordinates (the pullback metric, measured by how the data move, and the Fisher metric, measured by how the probability of an outcome changes)
- Why the Fisher metric measures length in only one direction when the only outcome is a single binary one (rank 1)
- Reading the direction where the two rulers disagree as a direction the data map overlooks
- Testing the functional form of the stability index Σ by comparing nested models (likelihood-ratio test)
- Python code (JAX) that checks these ideas on fictional data

Japanese only (for now). With code.

PDF of the article (Japanese, 99 pages): [fisher_metric_on_expert_coordinates_JA.pdf](./fisher_metric_on_expert_coordinates/fisher_metric_on_expert_coordinates_JA.pdf)

## Licence

CC BY 4.0.

## Contact

etalecohomology2026@proton.me

