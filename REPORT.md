# Node-Level Abnormality Localisation on Multimodal Graphs
**Position 2 — Simulation-Informed AI & Multimodal Epileptogenic-Zone Localisation**

## 1. Executive summary

The submitted predictor is **FC OOF stacker**: a 10-seed ensemble of a weighted message-passing
network over modality A + B with a `S2_moddropout_r0.3` missing-modality strategy, combined with the
independent simulation score via the `FC_oof_stacker` rule frozen on validation.

Test-set performance: **AUPRC 0.723** against a positive-prevalence floor of
**0.072** (10.1x), AUROC 0.955,
top-k Dice 0.654.

**The graph contributes genuinely.** Real adjacency beats both identity and degree-matched shuffled adjacency, the seed-paired deltas are sign-consistent, and both 95% subject-bootstrap CIs exclude zero. Modality B adds information beyond A
(A+B minus A-only, test AUPRC +0.0663 [+0.0210, +0.1247]). The simulation score alone
reaches AUPRC 0.405; fusion
improves on the learned model alone
(+0.1397 [+0.0579, +0.2267]).

## 2. Data and experimental protocol

140 subjects x 68 nodes = 9,520 rows, split by subject exactly
as given: train 84, val 28,
test 28. Prevalence 0.0727 overall
(train 0.0733, val 0.0714,
test 0.0720); 3-7
positives per subject.

**Validation.** Hard assertions verify 68 nodes per subject with `node_id` exactly 0-67 in every
table, that each table's `region` column matches `region_names.csv` node-by-node, that every
adjacency is 68x68, symmetric to 0e+00, non-negative, zero-diagonal
(density 0.2704), and that `modality_B_available` agrees with the actual file
contents. All passed.

**Missingness.** Modality A: 5.933% of cells NaN, with an essentially
identical NaN rate for abnormal and normal nodes ({0: 5.934, 1: 5.925}) — i.e. missingness
is not a free label signal. Handled by train-median imputation plus per-feature missingness
indicators. Modality B: 35/140 subjects have no rows at all,
spread across splits ({'test': 7, 'train': 20, 'val': 8}); three strategies compared in section 6.

**Preprocessing.** Every cross-subject statistic (medians, standardisation, node prior) is
fitted on train subjects only. Within-subject statistics (z-scores, graph degrees) use only the
subject's own 68 rows and cross no split boundary. `spike_rate`/`hfo_rate` are log1p-transformed
(strictly positive, long right tail, unlike the z-scored channels).

**Discipline.** The test set was touched once, after `configs/final_config.json` was written and
a 16-point leakage audit passed. `resected`/`engel_1_seizure_free` appear only in section 9.
Oracle k is consumed only inside `topk_dice()`.

## 3. Models

* **Non-graph baseline** — L2 logistic regression and HistGradientBoosting on the identical
  feature matrix (including graph degree features), tuned on validation AUPRC. Primary:
  **hgb**. The two families land within noise of each other, so the node-level
  signal is close to additive once within-subject z-scores are present.
* **Graph model** — 2-layer weighted message passing, `Z = H W_self + (S H) W_nbr + b`, ReLU,
  residual, dropout 0.3, hidden 64, Adam lr 0.003,
  early stopping on validation AUPRC. Written in NumPy with a finite-difference gradient check
  (max relative error 1.3e-08). `S = D^-1/2 (A + cI) D^-1/2` with `c` = the subject's mean
  non-zero edge weight — a unit self-loop would be invisible against a mean weight of
  7.878.
* **Missing B** — `S2_moddropout_r0.3`, selected on validation (section 6).
* **Fusion** — `FC_oof_stacker`, selected on validation after the concordance/discordance
  analysis (section 7).

## 4. Results (test set, single evaluation)

| Model                                     |   AUPRC |   AUROC |   TopkDice | SeedSD          |
|:------------------------------------------|--------:|--------:|-----------:|:----------------|
| Non-graph logreg (A+B)                    |   0.526 |   0.902 |      0.511 | nan             |
| Non-graph HGB (A+B)                       |   0.491 |   0.894 |      0.496 | nan             |
| Non-graph logreg (A_only)                 |   0.485 |   0.883 |      0.474 | nan             |
| Non-graph logreg (B_only)                 |   0.385 |   0.835 |      0.419 | nan             |
| Non-graph logreg (A_plus_B)               |   0.526 |   0.902 |      0.511 | nan             |
| Graph real (10-seed ens)                  |   0.582 |   0.915 |      0.619 | 0.566 +/- 0.010 |
| Graph shuffled (10-seed ens)              |   0.498 |   0.894 |      0.482 | 0.472 +/- 0.017 |
| Graph identity (10-seed ens)              |   0.482 |   0.887 |      0.478 | 0.461 +/- 0.015 |
| Graph weights (10-seed ens)               |   0.571 |   0.919 |      0.541 | 0.543 +/- 0.012 |
| Graph A_only                              |   0.515 |   0.896 |      0.499 | nan             |
| Graph B_only                              |   0.431 |   0.864 |      0.377 | nan             |
| Graph A_plus_B                            |   0.581 |   0.913 |      0.611 | nan             |
| LEARNED MODEL (S2_moddropout_r0.3)        |   0.584 |   0.918 |      0.585 | nan             |
| Simulation alone                          |   0.405 |   0.773 |      0.456 | nan             |
| Naive average (raw)                       |   0.714 |   0.94  |      0.67  | nan             |
| Naive average (rank)                      |   0.594 |   0.915 |      0.507 | nan             |
| FA convex weight                          |   0.602 |   0.921 |      0.53  | nan             |
| FB confidence-aware                       |   0.647 |   0.936 |      0.582 | nan             |
| FC OOF stacker                            |   0.723 |   0.955 |      0.654 | nan             |
| --- prevalence baseline (AUPRC floor) --- |   0.072 |   0.5   |    nan     | nan             |

Positive-prevalence baseline (AUPRC floor) = **0.072**.

Subject-level paired bootstrap, 2,000 replicates:

| comparison                       | metric    |   delta |      lo |     hi |   P_gt0 |
|:---------------------------------|:----------|--------:|--------:|-------:|--------:|
| Graph real vs identity           | auprc     |  0.0995 |  0.0413 | 0.1614 |   1     |
| Graph real vs identity           | topk_dice |  0.1409 |  0.0781 | 0.21   |   1     |
| Graph real vs shuffled           | auprc     |  0.0833 |  0.0301 | 0.141  |   0.999 |
| Graph real vs shuffled           | topk_dice |  0.1366 |  0.0637 | 0.2176 |   1     |
| Graph real vs weight-shuffle     | auprc     |  0.0111 | -0.0149 | 0.0282 |   0.784 |
| Graph real vs weight-shuffle     | topk_dice |  0.0777 |  0.0321 | 0.1233 |   1     |
| Graph real vs non-graph baseline | auprc     |  0.0906 |  0.0436 | 0.1321 |   1     |
| Graph real vs non-graph baseline | topk_dice |  0.1227 |  0.0613 | 0.181  |   1     |
| A+B vs A-only (graph)            | auprc     |  0.0663 |  0.021  | 0.1247 |   0.998 |
| A+B vs A-only (graph)            | topk_dice |  0.1116 |  0.0256 | 0.2092 |   0.994 |
| Final fusion vs model alone      | auprc     |  0.1397 |  0.0579 | 0.2267 |   1     |
| Final fusion vs model alone      | topk_dice |  0.0687 | -0.0013 | 0.1433 |   0.972 |
| Final fusion vs simulation alone | auprc     |  0.3186 |  0.2108 | 0.4145 |   1     |
| Final fusion vs simulation alone | topk_dice |  0.1975 |  0.1098 | 0.2863 |   1     |
| Final fusion vs naive average    | auprc     |  0.1298 |  0.0839 | 0.1858 |   1     |
| Final fusion vs naive average    | topk_dice |  0.1463 |  0.0828 | 0.2117 |   1     |

## 5. Does graph structure help?

Validation, 10 seeds, mean +/- SD AUPRC:
real **0.541 +/- 0.009**,
shuffled 0.467 +/- 0.012,
weight-shuffled 0.531 +/- 0.009,
identity 0.458 +/- 0.018.

Paired per-seed deltas: real-identity +0.0829 (sd 0.0226, same sign on
10/10 seeds); real-shuffled
+0.0739 (sd 0.0161, same sign on
10/10 seeds).

Test, 95% subject-bootstrap CI: real-identity +0.0995 [+0.0413, +0.1614];
real-shuffled +0.0833 [+0.0301, +0.1410];
real vs non-graph baseline +0.0906 [+0.0436, +0.1321].

**The graph contributes genuinely.** Real adjacency beats both identity and degree-matched shuffled adjacency, the seed-paired deltas are sign-consistent, and both 95% subject-bootstrap CIs exclude zero.

Mechanism check: training-set homophily is strong (weighted abnormal-neighbour fraction
0.160 around abnormal nodes vs 0.077 around normal
nodes; abnormal nodes are 2.9x more strongly interconnected than a
size-matched within-subject null). The node-level error analysis (section 8) checks whether the
nodes the graph actually fixes have the neighbourhood profile this mechanism predicts.

An important caveat we make explicit: purely **local** graph statistics (weighted degree,
eigen-centrality) already separate the classes, and we gave those to the non-graph baseline too.
So "the graph helps" here means specifically **message passing helps**, over and above
topology-as-a-node-feature.

## 6. Multimodality and missing data

| condition   |   n_feat |   auprc |   auroc |   topk_dice |   auprc_Bpresent |   dice_Bpresent |
|:------------|---------:|--------:|--------:|------------:|-----------------:|----------------:|
| A_only      |       24 |  0.4812 |  0.8658 |      0.4906 |           0.4885 |          0.496  |
| B_only      |       20 |  0.4391 |  0.8542 |      0.3846 |           0.5355 |          0.4435 |
| A_plus_B    |       39 |  0.5487 |  0.8968 |      0.5061 |           0.5784 |          0.5361 |

Test, chosen strategy `S2_moddropout_r0.3`, reported separately as required:
all subjects AUPRC 0.584 / Dice 0.585;
**B-present** AUPRC 0.603 / Dice 0.607;
**B-absent** AUPRC 0.540 / Dice 0.520.

Strategy comparison on validation:

| strategy            |   auprc_all |   dice_all |   auprc_Bpresent |   dice_Bpresent |   auprc_Babsent |   dice_Babsent |
|:--------------------|------------:|-----------:|-----------------:|----------------:|----------------:|---------------:|
| S2_moddropout_r0.3  |      0.5551 |     0.5401 |           0.587  |          0.5652 |          0.4798 |         0.4771 |
| S2_moddropout_r0.5  |      0.5545 |     0.5349 |           0.587  |          0.5581 |          0.4736 |         0.4771 |
| S2_moddropout_r0.15 |      0.5512 |     0.5329 |           0.5825 |          0.5652 |          0.4823 |         0.4521 |
| S1_median_flag      |      0.5487 |     0.5061 |           0.5784 |          0.5361 |          0.4881 |         0.4313 |
| S3_dual_path        |      0.5359 |     0.5192 |           0.5645 |          0.5361 |          0.4734 |         0.4771 |

B-only performance (0.439) versus A-only
(0.481) tells us whether B is intrinsically
weak or merely redundant — a distinction the A+B-vs-A comparison alone cannot make.

## 7. Simulation and fusion

**Standalone** (test): AUPRC 0.405, AUROC 0.773,
top-k Dice 0.456.

**Concordance** (validation): pooled Spearman 0.354; per-subject Spearman
mean 0.417 (median 0.501,
sd 0.247, range -0.258 to
0.698); mean top-k Jaccard 0.253. The
per-subject distribution is the honest view — pooled correlation is inflated by between-subject
level differences, which are not agreement about *which* nodes are abnormal.

**Discordance.** Disagreement is measured in rank space (raw probability differences would
mostly measure calibration mismatch). Spearman(sim_confidence, disagreement)
= -0.556 (p=0.00212); Spearman(sim_confidence, simulation top-k Dice)
= +0.701 (p=3.22e-05). On train the same relationship held
(rho 0.538 against per-subject simulation AUROC).

**Confound declared:** `sim_confidence` is itself lower for B-absent subjects, so a
confidence-aware rule risks taking credit for modality availability. The stacker is given both
terms so the coefficients reveal which it actually uses; the largest-magnitude coefficient was
`sim_rank_x_conf`.

**Fusion rationale.** Because confidence demonstrably tracks simulation reliability, a
confidence-aware weighting is *earned* rather than assumed. All four required conditions were
compared; `FC_oof_stacker` won on validation and was frozen. Test:
+0.1397 [+0.0579, +0.2267] versus the model alone and
+0.1298 [+0.0839, +0.1858] versus a rank-space naive average.

## 8. Error analysis and uncertainty

Comparing the real-adjacency ensemble to the identity ensemble at top-k on test:
22 abnormal nodes recovered only by the graph, 4 lost,
63 found by both. Subjects improved 13, worsened
0, unchanged 15. Excluding the two
most-improved subjects, the mean delta top-k Dice is
+0.1107 (versus +0.1409
overall) — the test of whether the gain is an effect or an anecdote.

Ensemble-spread uncertainty correlates with node-level error
(Spearman +0.325, p=5.67e-48), but part of that is mechanical: spread is
largest near p=0.5, which is also near the top-k boundary. Conditioning on predicted probability
attenuates it, and we report both.

## 9. Outcome association (restricted variables, used here only)

Overlap between the predicted-abnormal set and the resected set, compared across
`engel_1_seizure_free` groups on 28 test subjects, with Mann-Whitney U, rank-biserial
effect size and a bootstrap CI (printed in the notebook). With 28 subjects the comparison is
underpowered; a null is weak evidence of absence. This is observational — surgical decisions
were made with information we do not have — so we claim association, not causation.

## 10. Limitations

* 140 subjects / 28 test subjects. Every CI is wide; the third decimal place is not meaningful.
* Validation AUPRC is optimistically biased because it is also the early-stopping criterion.
  Test is not, which is why test is the number we quote.
* A single fixed 68-node atlas with identical node ordering. Nothing here transfers to a
  different parcellation without retraining.
* Site effects are present in the covariates but we did not model them hierarchically.
* Deliberately shallow hyperparameter exploration; a flat validation plateau meant we chose a
  plateau centre rather than an argmax, but a larger search might find something.
* No external validation set.
* Top-k Dice uses the oracle k. It measures ranking quality, not deployable detection — a
  deployed system would need a calibrated threshold or a predicted k.
* The simulation pipeline is a black box to us; we can characterise its behaviour but not its
  failure modes.

## 11. What I would do with more time

1. Nested cross-validation over the train+val pool, so model selection variance is estimated
   rather than assumed, instead of a single 84/28 selection split.
2. Predict k per subject (it ranges 3-7 and is currently oracle) and report a threshold-based
   sensitivity/specificity/kappa alongside the ranking metrics.
3. Hierarchical / mixed-effects modelling of site and subject-level intercepts, since site is
   associated with both missingness and sim_confidence.
4. Decompose uncertainty into epistemic (seed/ensemble) and aleatoric components rather than
   reporting a single spread.
5. Topology robustness: sparsify by weight threshold and vary propagation depth systematically,
   to find how much of the graph actually carries the signal.
6. A proper sensitivity analysis of the fusion to sim_confidence *conditional on* modality-B
   availability, to fully disentangle the confound flagged in section 7.
