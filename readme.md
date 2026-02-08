# MADE-PPE: Multi-agent Automated Debate for Explainable Labeling and Schema Restructuring

MADE-PPE builds a regulation-grounded benchmark that links work situations and hazard factors to required PPE, then evaluates wearing and improper-wearing for compliance assessment.
It uses a multi-agent pipeline (proposer–rebutter–judge) to produce structured labels and iteratively refine the schema via aggregation and statistical re-structuring.

## Figures

### Fig. 1. MADE-PPE multi-agent labeling workflow
![Fig. 1. MADE-PPE multi-agent labeling workflow](.git_information/fig_1.jpg)

**Description.** Given an input sample, agents (Proposer–Rebutter–Judge) sequentially infer **work situation → hazard factors → compliance**, and then estimate **wearing** and **improper-wearing**. The final output is stored as a structured record, and optional schema-update proposals are aggregated for refinement.

### Fig. 2. End-to-end research pipeline
![Fig. 2. End-to-end research pipeline](.git_information/fig_2.jpg)

**Description.** Starting from PPE datasets, the pipeline defines a **regulation-based pre-structure (work environment–hazards–PPE)**, performs **multi-agent labeling**, conducts **statistical re-structuring**, and finally runs **benchmark tests** (e.g., SOTA VLM-based evaluation).

## Run

```bash
python -m run
```

## Benchmark run

```bash
python -m build_train_jsonl
```

```bash
python -m train_vlm
```

```bash
python -m eval_vlm
```
