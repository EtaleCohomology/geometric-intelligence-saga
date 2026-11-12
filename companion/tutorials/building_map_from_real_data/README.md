[English](./README.md) | [日本語](./README.ja.md)

# Building a "Map" from Real Socioeconomic Data

Real socioeconomic data are messy: missing values, outliers, mixed units and magnitudes, and mixed time granularities. This introductory tutorial walks through selecting data, treating these problems, and building a map (a VAE latent space) that passes the three conditions. This folder contains the code for the fourth article in the Qiita series "Geometric Intelligence".

## Target Audience

- Python machine learning engineers working in companies and government agencies
- Management consultants
- Policy consultants
- Policy staff and secretaries of members of parliament who are familiar with machine learning (AI) and AI agents
- Researchers and research scientists interested in applying advanced mathematics as practical tools for business and policy analysis

## Contents

- Choosing data backwards from the question the map should answer
- Real-data problems (record linkage, granularity, units, missing values, outliers, reporting lags and revisions, structural change, data that cannot leave an organisation) and how to treat them
- Diagnosing which of the three conditions failed, and how to fix it
- Proposed map designs by industry (finance, manufacturing, energy)
- How to keep a record of trial and error
- Python code (JAX) that turns a shape with a known answer into deliberately messy data and checks how different preprocessing changes the three conditions

## Code

- [code/gi_real_data_preprocessing_jax.py](./code/gi_real_data_preprocessing_jax.py)

```
pip install -r code/requirements.txt
python code/gi_real_data_preprocessing_jax.py clean 0.05               # reference: undisturbed data
python code/gi_real_data_preprocessing_jax.py naive 0.05               # (A) fill missing values with 0
python code/gi_real_data_preprocessing_jax.py zscore 0.05              # (B) standardise only
python code/gi_real_data_preprocessing_jax.py careful 0.05             # (C) full treatment
python code/gi_real_data_preprocessing_jax.py careful_mask 0.05        # (D) (C) + missing-value indicators
python code/gi_real_data_preprocessing_jax.py careful_quarterly 0.05   # (E) (D) + aligned granularity
python code/gi_real_data_preprocessing_jax.py careful_quarterly 0.05 3 # optional 3rd argument: training random seed (default 1)
```

The numbers in the article were obtained by running this code in the author's environment.

## Suggested Reading Order

1. [Part 1](../vae_to_socioeconomic_map/) — checking the map (may a ruler be laid on it?)
2. [Part 2](../reading_current_position_shape/) — reading the shape of where we stand now
3. [Part 3](../deciding_how_to_move/) — deciding how to move
4. Part 4 (this tutorial) — building a map from real data
5. [gi_theory_introduction_JA.pdf](../gi_theory_introduction_JA.pdf) — a systematic introduction to GI theory

## Related Material

- Data constraints by industry and policy domain, and the trial and error needed to pass the conditions (fictional case studies): [gi_practice_materials](../gi_practice_materials/)
- The fictional settings discussed in the article (case study novels by industry and policy domain): [case-studies](../../../case-studies/)

## Note

The case study content introduced in the article consists of fictional story settings written by the author. It does not describe findings obtained from real data of real companies or government agencies. The industry map designs are the author's proposals and do not indicate use at any real company.

## License

CC BY 4.0
