[English](./README.md) | [日本語](./README.ja.md)

# GI Theory Practice Materials

Teaching materials that use the GI case study novels (by industry and policy domain) to organise the workflow of building a manifold in practice, analysing it, and demonstrating that the pipeline is correct.

## Purpose

In the case study novels, the scenes in which a differentiable manifold is built from data, analysed, and shown to be correct to stakeholders are spread across the story. These materials organise those scenes across the works, so that readers can learn the reasoning and procedures needed to do the same work in practice.

## Contents

For first-time readers, we recommend the following order: 01 → 02 (as needed) → 03 → 04

### `01_manifold_quality_checklist_JA.pdf`

*Building a differentiable manifold from data: the trial and error needed to pass the quality checklist.*

For seven case studies, this document organises, in text and tables, which condition of the quality checklist (the three conditions of Proposition 1.1) failed first, why, and how it was eventually passed. Japanese only (for now). Approximately 21 pages.

### `02_manifold_quality_checklist_detailed_JA.pdf`

*(Detailed edition) Industry-specific data constraints and the trial and error needed to pass the quality checklist.*

A detailed edition of 01. It covers the characteristics and constraints of the datasets in each industry and policy domain (separately managed projects, fragmented core systems, cyclical structure, confidentiality walls between customers, inputs restricted to aggregated indicators, etc.), and the steps taken to overcome them and pass the checklist. Japanese only (for now). Approximately 20 pages.

### `03_analyses_on_the_manifold_JA.pdf`

*What was analysed on the finished manifold.*

This document organises the analyses and simulations carried out in each case study on a manifold that passed the quality checklist (scalar curvature, Lie derivatives, Lie brackets, geodesics and optimal control, persistent homology, surgery theory, symplectic integration, sheaves, etc.). Methods not covered in the GI theory papers and companion volumes are also organised by work. Japanese only (for now). Approximately 16 pages.

### `04_verification_and_accountability_JA.pdf`

*Demonstrating that the pipeline is correct without revealing confidential information.*

This document organises what was done in each case study to show boards, governments, oversight bodies and parliaments that the pipeline works as promised (formal verification, theorem proving, zero-knowledge proofs, differential privacy, federated learning, secret sharing, etc.), together with what each method does and does not demonstrate. Japanese only (for now). Approximately 12 pages.

## Notes

- Each document is organised within the scope of what is written in the case study manuscripts. Numbers not written in the manuscripts have not been filled in by guesswork.
- The numbers in the manuscripts are fictional values from the stories, not results computed on real data. All companies, organisations, people and data are fictional.
- Some documents also refer to the forthcoming work "The Enemy's Manifold" (China edition). Descriptions may change in the published version.

## Related Material

- Introduction: [vae_to_socioeconomic_map](../vae_to_socioeconomic_map/)
- Case study novels: [case-studies](../../../case-studies/)

## Licence

CC BY 4.0
