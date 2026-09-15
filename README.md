# PRS1 — IIIT-Delhi Take-Home Assignment

**Candidate:** Lighittha P R  
**Email:** ligpersonalmail@gmail.com


This repository contains my complete solution for **Position 2 — Simulation-Informed AI & Multimodal Epileptogenic-Zone Localisation**. In the application Google Form, I can separately select my preferred project position; this repository answers the take-home brief itself.

## Repository structure

```text
repo-root/
├── src/
│   └── pipeline.py              # end-to-end code, ordered by assignment Sections 1-7
├── data/
│   └── README.md                # where to place the provided dataset
├── ASSIGNMENT.ipynb             # narrative notebook version of the same analysis
├── predictions.csv              # 1,904 final test predictions
├── requirements.txt             # exact pinned CPU dependencies
├── README.md
├── REPORT.md                    # compact report following Sections 1-7
├── ANALYSIS.md                  # required graph-contribution answer
├── configs/
│   └── final_config.json        # configuration frozen before predictive test evaluation
└── outputs/                     # supporting tables, figures, audit files and experiment ledger
```

## Result-file conventions

The primary report and prediction file contain no accidental `NaN` metric values. A few supporting CSVs intentionally contain blank cells where a field is **not applicable** (for example, seed-to-seed SD for a deterministic or derived fusion, or an optional audit-detail cell). These blanks do not represent failed runs or missing predictions. Missing modality-A values in the supplied dataset are genuine input missingness and are handled by the pipeline as described below.

## Setup

I use Python 3.10+ and only NumPy/scikit-learn/scipy/pandas/matplotlib code. The custom message-passing model is implemented directly in NumPy; **no GPU and no deep-learning framework are required**.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the supplied dataset under `data/` as described in `data/README.md`.

## Run end to end

From the repository root:

```bash
python src/pipeline.py
```

The full robustness suite is intentionally extensive (main ablations, repeated seeds, capacity/design sensitivity, modality experiments, OOF stacking, CV stability and bootstrap analyses). On the environment used for the final run, the complete pipeline required roughly **4-6 CPU hours**. The Kaggle session had accelerators attached, but the submitted code contains no CUDA/GPU path; computation is NumPy/scikit-learn CPU work.

The notebook is included for readability, but `src/pipeline.py` is the canonical one-command execution path. Seeds are explicit (`SEED=42`) and the graph conditions use deterministic seed schedules.

## Final submitted model

I freeze an **OOF stacked fusion** on validation. Its learned-model component is a 10-seed ensemble of the two-layer weighted message-passing network using the selected modality-dropout strategy (`r=0.3`), and the stacker combines learned-model and independent simulation evidence.

Final test performance:

- **AUPRC:** 0.722 (prevalence baseline 0.072)
- **AUROC:** 0.954
- **Top-k Dice:** 0.659

I do not claim the stacker is proven better than the simple raw average on test: raw 50/50 averaging reaches AUPRC 0.714 and Dice 0.670, and the test CI for the AUPRC difference crosses zero. I keep the stacker because it was the validation-selected model before test evaluation.

## Main conclusions

- **Graph structure contributes:** real adjacency 0.582 AUPRC versus shuffled 0.498 and identity 0.482 on test; subject-bootstrap CIs for both required contrasts exclude zero.
- **Topology matters more than exact weights:** weight-shuffled topology reaches 0.571 AUPRC and the real-minus-weight-shuffle AUPRC CI crosses zero.
- **Modality B adds information:** A+B beats A-only by +0.0663 AUPRC [0.0210, 0.1247].
- **Missing B is handled explicitly:** the selected learned model scores 0.603 AUPRC for B-present and 0.540 for B-absent test subjects.
- **Simulation is complementary:** standalone AUPRC is 0.405; `sim_confidence` predicts simulation reliability/disagreement and motivates confidence-aware fusion.

## Test-set protocol and an important wording clarification

I use train subjects for fitting and validation subjects for all model, hyperparameter, missing-modality and fusion selection. The final predictive test metrics and final prediction file are produced only after the configuration is frozen.

Phase A performs dataset-integrity/descriptive checks over the provided files (for example split sizes and missingness summaries). Those checks do not fit parameters or choose models. I therefore do **not** describe the repository as literally never reading any test-file value before freeze; the defensible claim is that **test predictive performance and test labels never drive training or selection**.

Restricted variables (`resected`, `engel_1_seizure_free`) are isolated from modelling and appear only in the optional post-hoc outcome analysis. True per-subject `k` is used only for evaluation/diagnostic top-k calculations, never for training, fusion, thresholding or `predictions.csv`.

## Assumptions and tradeoffs

1. **Adjacency ordering:** the `.npy` matrices contain no node index, so I cannot directly prove row/column order. I document the assumption and provide indirect hemispheric/homotopic evidence in `outputs/adjacency_ordering_evidence.json`.
2. **Hemisphere covariate:** `subjects.hemisphere` is not restricted by the brief, so I use an `is_ipsilateral` feature. Because this may represent clinical-workup information in deployment, I also report a no-clinical-prior sensitivity result.
3. **Validation reuse:** validation is heavily reused for selection and is not treated as an unbiased performance estimate. Test is the performance split.
4. **Unequal ensemble sizes:** the main graph ablations use 10 seeds; modality arms use 5 and non-graph baselines are single fits. I therefore base the graph-structure claim primarily on seed-matched real/shuffled/identity controls, not graph-versus-baseline alone.
5. **Simulation provenance:** I can characterise `sim_score` but cannot inspect the producing pipeline, so I avoid causal/provenance claims about it.

## Prediction-file checks

`predictions.csv` has exactly:

```text
subject_id,node_id,prob_abnormal
```

It contains 1,904 rows = 28 test subjects × 68 nodes, each node 0-67 exactly once per subject, no duplicate keys, no NaNs, and probabilities in [0,1].

## Supporting outputs

`outputs/` contains the complete master table, graph seed/ablation tables, subject-bootstrap CIs, modality/missingness comparisons, concordance/discordance analyses, capacity/design sensitivities, calibration, uncertainty/error analyses, optional outcome analysis, and figures. `ANALYSIS.md` gives the required focused answer on whether the graph contribution is genuine.
