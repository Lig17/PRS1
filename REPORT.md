# REPORT.md — Node-Level Abnormality Localisation on Multimodal Graphs
**Position 2 — Simulation-Informed AI & Multimodal Epileptogenic-Zone Localisation**

**Candidate:** Lighittha P R  
**Email:** ligpersonalmail@gmail.com


## 1. Data Handling and Setup

I model 140 independent subjects with 68 aligned nodes each (9,520 node rows), using the provided subject split: 84 train, 28 validation, and 28 test. The positive prevalence is 0.073 on train and 0.072 on test, so I use **AUPRC as the primary metric**, always alongside the prevalence floor, plus pooled AUROC and per-subject oracle-k Dice exactly as requested. I do not report accuracy.

I align every table by `(subject_id, node_id)` rather than row order, verify 68 nodes per subject and `node_id=0..67`, check region-name consistency, and validate every adjacency for shape, symmetry, non-negativity and zero diagonal. The `.npy` adjacency files do not contain node labels, so exact adjacency ordering cannot be proven from the files alone; I state this as an assumption. The observed hemispheric block structure and strong homotopic connections are consistent with the supplied node order, and I save that evidence in `outputs/adjacency_ordering_evidence.json`.

Modality A has 5.933% missing cells. I use train-fitted median imputation, per-feature missingness flags, and within-subject z-scores. Modality B is absent entirely for 35/140 subjects. All train-fitted cross-subject statistics are frozen before evaluation. The restricted variables `resected` and `engel_1_seizure_free` are excluded from training and are used only in the optional outcome analysis.

## 2. Baseline Model (Non-Graph)

I tune two non-graph families on validation: L2 logistic regression and HistGradientBoosting. Both receive the same learned-node feature matrix used by the graph model, including local graph statistics such as degree and eigen-centrality, so a graph gain cannot be explained merely by giving the GNN access to topology-derived node covariates. HGB is the validation-selected primary baseline; logistic regression is also reported because it is stronger on the test split.

## 3. Graph Model and Ablations

I use one custom two-layer weighted message-passing model implemented in NumPy. The layer is `Z = H W_self + (S H) W_nbr + b`, with ReLU, residual connection, dropout 0.3 and hidden width 64. I normalise adjacency as `S = D^-1/2 (A + cI) D^-1/2`, where `c` is each subject's mean non-zero edge weight. A finite-difference gradient check gives maximum relative error `1.3e-08`.

The required graph ablations are decisive. On validation across 10 paired seeds, AUPRC is **0.541±0.009 real**, **0.467±0.012 shuffled**, and **0.458±0.018 identity**; real-minus-shuffled and real-minus-identity have the same sign on 10/10 seeds. On test, the 10-seed ensembles score **0.582 real**, **0.498 shuffled**, and **0.482 identity**. Subject-bootstrap CIs exclude zero for real-vs-shuffled and real-vs-identity. Weight shuffling while keeping topology gives 0.571, and its AUPRC difference from real is only +0.0111 with a CI crossing zero. My interpretation is therefore narrow: **correct topology contributes substantially, while exact edge-weight values add much less**.

| comparison                          | 95% subject-bootstrap CI   |
|:------------------------------------|:---------------------------|
| Graph real vs identity              | +0.0995 [+0.0413, +0.1614] |
| Graph real vs shuffled              | +0.0833 [+0.0301, +0.1410] |
| Graph real vs weight-shuffle        | +0.0111 [-0.0149, +0.0282] |
| Graph real vs non-graph HGB         | +0.0906 [+0.0436, +0.1321] |
| Graph real vs non-graph logreg      | +0.0561 [-0.0030, +0.1159] |
| A+B vs A-only (graph)               | +0.0663 [+0.0210, +0.1247] |
| Final fusion vs model alone         | +0.1379 [+0.0564, +0.2242] |
| Final fusion vs naive average (raw) | +0.0080 [-0.0327, +0.0576] |

I also rerun real/shuffled/identity at small, frozen and large capacities; real remains best at all three, so the graph gap is not tied to one width/depth choice.

## 4. Multimodal Fusion and Missing Modalities

The test graph arms show that modality B adds information beyond A: **A-only 0.515 AUPRC, B-only 0.431, A+B 0.581**. The A+B versus A-only subject-bootstrap difference is **+0.0663 [0.0210, 0.1247]**. These main arms include the common covariates (`is_ipsilateral` and local graph statistics); I also provide isolated modality checks in `outputs/table_modality_isolated.csv` so the incremental and standalone views are both available.

For missing B, I compare median+flag, three modality-dropout rates, and a dual-path model on validation. I select **modality dropout at r=0.3**. On test, the selected learned model scores **0.584 AUPRC / 0.585 Dice overall**, **0.603 / 0.607 for B-present subjects**, and **0.540 / 0.520 for B-absent subjects**. Thus the strategy handles fully missing B without dropping subjects, while retaining strong performance in the B-present majority.

## 5. Independent Simulation Score and Fusion

I first evaluate `sim_score` independently as required: on test it reaches **AUPRC 0.405, AUROC 0.773, Dice 0.456**. On validation, model/simulation concordance is moderate (pooled Spearman 0.354; mean per-subject Spearman 0.417; mean top-k Jaccard 0.253). Discordance is informative: `sim_confidence` correlates negatively with disagreement (Spearman -0.556, p=0.0021) and positively with simulation Dice (+0.701, p=3.2e-05). I therefore test confidence-aware fusion rather than assuming it a priori.

I compare more than the four required conditions: model alone, simulation alone, raw 50/50 average, rank-space average, validation-tuned convex weighting, confidence-aware weighting, and an OOF stacker. The OOF stacker is trained only on subject-level out-of-fold train predictions with a separate inner early-stopping split, and is selected on validation (**0.720 AUPRC**) before test evaluation.

On test, the frozen stacker reaches **AUPRC 0.722, AUROC 0.954, Dice 0.659**. It clearly improves over the learned model alone (+0.1379 AUPRC, 95% CI [0.0564, 0.2242]), but the simple raw average is already very strong at **0.714 AUPRC / 0.670 Dice**. The stacker-minus-raw-average AUPRC CI crosses zero (+0.0080 [-0.0327, 0.0576]), and its Dice is slightly lower. I therefore do **not** claim that the complex fusion is proven superior to the naive raw average; I retain it only because it was the validation-selected final model, as required by the protocol.

## 6. Evaluation and Reporting

Positive-prevalence baseline on test: **0.072**. The following table covers every reported model from Sections 2-5.

| Model                                            | Seeds   |   AUPRC |   AUROC |   TopkDice |
|:-------------------------------------------------|:--------|--------:|--------:|-----------:|
| Non-graph logreg (A+B)                           | 1       |   0.526 |   0.902 |      0.511 |
| Non-graph HGB (A+B)                              | 1       |   0.491 |   0.894 |      0.496 |
| Non-graph logreg (A_only)                        | 1       |   0.485 |   0.883 |      0.474 |
| Non-graph logreg (B_only)                        | 1       |   0.385 |   0.835 |      0.419 |
| Non-graph logreg (A_plus_B)                      | 1       |   0.526 |   0.902 |      0.511 |
| Graph real (10-seed ens)                         | 10      |   0.582 |   0.915 |      0.619 |
| Graph shuffled (10-seed ens)                     | 10      |   0.498 |   0.894 |      0.482 |
| Graph identity (10-seed ens)                     | 10      |   0.482 |   0.887 |      0.478 |
| Graph weights (10-seed ens)                      | 10      |   0.571 |   0.919 |      0.541 |
| Graph A + shared covariates                      | 5       |   0.515 |   0.896 |      0.499 |
| Graph B + shared covariates                      | 5       |   0.431 |   0.864 |      0.377 |
| Graph A+B + shared covariates                    | 5       |   0.581 |   0.913 |      0.611 |
| LEARNED MODEL (S2_moddropout_r0.3)               | 10      |   0.584 |   0.918 |      0.585 |
| Simulation alone                                 | 0       |   0.405 |   0.773 |      0.456 |
| Simulation alone (within-subject rank transform) | 0       |   0.365 |   0.816 |      0.456 |
| Naive average (raw)                              | derived |   0.714 |   0.94  |      0.67  |
| Naive average (rank)                             | derived |   0.594 |   0.915 |      0.507 |
| FA convex weight                                 | derived |   0.602 |   0.921 |      0.53  |
| FB confidence-aware                              | derived |   0.647 |   0.936 |      0.582 |
| FC OOF stacker                                   | derived |   0.722 |   0.954 |      0.659 |

The submitted `predictions.csv` contains exactly 1,904 rows and exactly the required columns: `subject_id,node_id,prob_abnormal`; all 28 test subjects have nodes 0-67 once each, there are no NaNs or duplicate keys, and every probability is in [0,1]. The submitted fusion is also well calibrated descriptively on test (Brier 0.0352, ECE 0.0106), although calibration statistics were not used for model selection.

## 7. Robustness, Limitations, and What I Would Do Next

I use 10 seeds for the main graph ablations, paired seed-wise differences, 2,000-replicate subject-level bootstrap CIs, a three-capacity sensitivity analysis, design-choice ablations, and fixed-config 5-fold train+validation stability analysis (AUPRC 0.555±0.026). Error analysis finds 22 positive nodes recovered only by real message passing versus 4 lost; 13/28 subjects improve and none worsen relative to identity at top-k. However, the proposed homophily mechanism is **underpowered** at node level: fixed and harmed nodes do not show a resolved difference in abnormal-neighbour connectivity, so I do not use that mechanism test as positive evidence.

The main limitations are the small 28-subject test set, heavy reuse of the validation split for selection, unequal ensemble sizes across some table rows, assumed rather than directly verifiable adjacency ordering, potential deployment circularity of `subjects.hemisphere`, lack of an external validation cohort, and uncertainty about the provenance of the provided simulation score. Top-k Dice also uses true k only as an evaluation oracle and is not itself a deployable thresholding rule.

With more time, I would run fully nested cross-validation so selection variance is estimated rather than only fit variance, predict k or tune a deployable threshold, model site effects hierarchically, test graph sparsification/edge-threshold robustness, and evaluate the fusion-confidence relationship conditional on modality-B availability.
