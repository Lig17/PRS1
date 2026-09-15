# Node-Level Abnormality Localisation on Multimodal Graphs (Position 2)

Predicts, for every node of every subject, the probability that the node is abnormal, and
answers the four required analysis questions.

## What is here
```
ASSIGNMENT.ipynb   run top-to-bottom; produces everything below
predictions.csv    1,904 rows (28 test subjects x 68 nodes), 3 columns
REPORT.md          <=5 pages, auto-generated from executed results
ANALYSIS.md        the "is the graph genuinely contributing?" answer
configs/           frozen final_config.json, written BEFORE the test set is touched
outputs/           all tables (.csv), figures (.png), audit, experiment ledger
```

## Environment
Python 3.10+. `pip install -r requirements.txt`. **CPU only; no GPU, no deep-learning
framework.** The message-passing layer is hand-written NumPy with a finite-difference gradient
check that runs in the notebook.

## Data layout expected
```
data/
  subjects.csv  modality_A.csv  modality_B.csv
  simulation_scores.csv  node_labels.csv  region_names.csv
  adjacency/<subject_id>_adj.npy      # 140 files, float32 68x68
```
The notebook auto-detects `data/`, `candidate_package/data/`, or any Kaggle input path
containing `subjects.csv` next to an `adjacency/` directory.

## Reproduce
```
jupyter nbconvert --to notebook --execute ASSIGNMENT.ipynb --output executed.ipynb
```
Runtime ~25-45 minutes on a laptop CPU (dominated by the 40 ablation runs and the
out-of-fold stacker). Seeds are fixed (`SEED = 42`); NumPy/sklearn are seeded and the
NumPy model is fully deterministic given a seed.

## Main modelling choices
* **Features.** Modality A (median-imputed) + per-feature missingness indicators + within-subject
  z-scores; modality B (log1p on the rate channels) + `B_available`; `is_ipsilateral` derived
  from `subjects.hemisphere`; four within-subject graph statistics. Node-identity prior tested
  and **rejected** on validation.
* **Graph.** `S = D^-1/2 (A + cI) D^-1/2`, `c` = the subject's mean non-zero edge weight (a unit
  self-loop would be invisible against a mean weight of ~7.878).
  2 layers, hidden 64, residual, dropout 0.3.
* **Missing B:** `S2_moddropout_r0.3` (three strategies compared, reported separately for B-present and
  B-absent subjects).
* **Fusion:** `FC_oof_stacker`, chosen only after standalone/concordance/discordance analysis.

## Assumptions (documented rather than hidden)
1. `subjects.hemisphere` is treated as a legitimate, non-restricted model input. It is not on the
   restricted list and it is available at prediction time.
2. Within-subject statistics are treated as leakage-free: they use only the subject's own 68 rows
   and would be computable at inference for a single isolated subject.
3. The final model is refit on **train only**, not train+val, so the early-stopping epoch remains
   meaningful and the reported test number is cleanly interpretable.
4. Ties in top-k selection are broken by stable sort order.
5. `sim_confidence` is a property of the simulation pipeline, not of the target, so it is used
   only in fusion — never as a node feature for the learned model.

## Reproducibility notes
`np.random.default_rng` with explicit seeds throughout; the adjacency permutation for the
shuffled ablation is seeded per (condition, seed) so it is reproducible and paired.

## >>> TEST-SET WARNING <<<
The test split is used **exactly once**, in Phase M, after `configs/final_config.json` is written
and a 16-point leakage audit passes (`TEST_UNLOCKED` is a module-level flag asserted by every
function that can read test data). No architecture, hyperparameter, threshold, fusion rule,
calibration map or ensemble decision was chosen using test results.
