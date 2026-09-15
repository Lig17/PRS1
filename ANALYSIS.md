# ANALYSIS.md

## "Your graph model beats your non-graph baseline by 1%. How would you determine whether the graph structure is genuinely contributing, rather than the difference being noise or an artefact of your setup?"

A single number beating another single number is not evidence. A 1% gap on 28 subjects is
comfortably inside the range that seed variance, a slightly better-regularised architecture, or
one lucky subject can produce. Below is what we actually ran, in the order that each step rules
out a specific alternative explanation.

### Step 1 — Rule out "it is just the architecture"  → IDENTITY adjacency

Set `S = I`. Because our layer keeps `W_self` and `W_nbr` separate, `S = I` turns it into an
exact dense MLP layer with **identical parameter count, optimiser, initialisation scheme,
training budget and early-stopping rule**. Anything the graph model gains over this is not
explained by "neural nets are better than logistic regression".

Observed (validation, 10 seeds): real 0.5413 +/- 0.0090
vs identity 0.4584 +/- 0.0180.
Paired per-seed delta +0.0829 (sd 0.0226),
same sign on 10/10 seeds.
Test, 95% subject-bootstrap CI: **+0.0995 [+0.0413, +0.1614]**.

### Step 2 — Rule out "any graph would do"  → SHUFFLED adjacency

This is the contrast that actually matters, and the one most submissions omit. We apply
`P A Pᵀ` with a random permutation while leaving the feature rows and labels in their original
node order. This **preserves** the weighted degree sequence, the density, the edge-weight
distribution and every global structural statistic, and **destroys only** the correspondence
between topology and features.

Why this is the decisive control: a random graph smooths, regularises and averages just as much
as the real one. If the gain were really "message passing is a nice regulariser on a small noisy
dataset", shuffled would match real. Only a gain that *disappears* under shuffling is a
statement about **which nodes are connected to which**.

Observed: real 0.5413 vs shuffled 0.4674,
paired delta +0.0739 (sd 0.0161), same sign on
10/10 seeds.
Test CI: **+0.0833 [+0.0301, +0.1410]**.

### Step 3 — Localise *what* about the graph matters  → WEIGHT-shuffled adjacency

Keep the binary topology, permute the non-zero weights. This separates "which edges exist" from
"how strongly they are weighted". Real vs weight-shuffled on test:
**+0.0111 [-0.0149, +0.0282]**. Validation: real 0.5413 vs
weight-shuffled 0.5310.

### Step 4 — Rule out "it is one seed"  → repeated seeds and paired deltas

10 seeds per condition. We report mean +/- SD, and we take deltas **paired by seed**,
so the two runs being compared share an initialisation and a dropout stream and the adjacency is
the only difference. The cheapest honest robustness check is **sign consistency**: a gain that
flips sign on some seeds is noise however good its mean looks.

### Step 5 — Rule out "it is one subject"  → subject-level bootstrap and per-subject deltas

Nodes inside a subject share a graph, a noise level and a simulation confidence, so they are not
independent. Treating ~1,900 test nodes as IID would give CIs that are far too narrow. Every CI
in this submission resamples **28 subjects with replacement** and takes all 68 nodes of each,
2,000 replicates, fixed seed; replicas are relabelled so a twice-drawn subject contributes twice
to the top-k Dice average.

We also report the per-subject delta distribution: 13 subjects improved,
0 worsened, 15 unchanged. Removing the two
most-improved subjects moves the mean delta top-k Dice from +0.1409 to
+0.1107.

### Step 6 — Rule out "the baseline was weak"  → fair comparison

The baseline gets the **same** feature matrix — including the graph-derived node statistics
(weighted degree, binary degree, mean edge weight, eigen-centrality), which our EDA showed
already separate the classes on their own. It is tuned on validation across two model families
(regularised logistic regression and gradient boosting) over a grid whose whole range was
0.460-0.508. Consequently our claim is narrow and specific:
**message passing helps beyond topology-as-a-node-feature.** Without giving the baseline the
degree features, a "graph helps" claim would be confounded by something a non-graph model can
trivially compute.

### Step 7 — Demand a mechanism  → node-level error analysis

A real effect should be explicable. Our EDA predicted the mechanism *before* the modelling:
abnormal nodes are 2.9x more strongly interconnected than a
size-matched within-subject null (Wilcoxon p=1.7e-15), and the weighted
fraction of abnormal neighbours is 0.160 around abnormal nodes versus
0.077 around normal ones. If message passing helps, it should help
*specifically* on nodes with abnormal neighbours.

So we compared the real and identity ensembles node by node at top-k: 22
positives recovered only by the graph, 4 lost. `table_error_analysis.csv`
contrasts the neighbourhood profile of fixed versus harmed nodes. If fixed nodes do **not** show
more/stronger connections to other abnormal nodes than harmed nodes, then the gain is not coming
from the mechanism we proposed and we should distrust it even if the CI looks good.

### Step 8 — Sensitivity to the setup

The conclusion should not hinge on one configuration. We varied depth (1/2/3), width, dropout,
learning rate, class weighting and the residual connection, and adopted a non-default setting
only if it beat the default by more than the default's own seed-to-seed spread. We also verified
the hand-written backward pass by finite differences (max relative error 1.3e-08) — an
unverified custom gradient is itself a plausible artefact.

---

## The falsification criterion, stated in advance

> We would conclude the graph structure does **not** genuinely contribute if **any** of:
> 1. real ≈ identity (95% subject-bootstrap CI on the difference includes 0);
> 2. real ≈ shuffled (CI includes 0) — i.e. a degree-matched random graph does just as well;
> 3. the sign of the paired per-seed delta flips across seeds;
> 4. the gain vanishes when the two most-improved subjects are removed;
> 5. graph-fixed nodes show no more connectivity to abnormal neighbours than graph-harmed nodes.

## Do our results meet it?

| Criterion | Result | Passed? |
|---|---|---|
| real − identity CI excludes 0 | +0.0995 [+0.0413, +0.1614] | YES |
| real − shuffled CI excludes 0 | +0.0833 [+0.0301, +0.1410] | YES |
| sign-consistent across seeds (real−identity) | 10/10 | YES |
| sign-consistent across seeds (real−shuffled) | 10/10 | YES |
| gain survives dropping 2 best subjects | +0.1107 vs +0.1409 | YES |

**Verdict.** **The graph contributes genuinely.** Real adjacency beats both identity and degree-matched shuffled adjacency, the seed-paired deltas are sign-consistent, and both 95% subject-bootstrap CIs exclude zero.

What we do **not** claim: that this generalises to another parcellation, that the edges are
causal, or that the effect size is precisely estimated. With 28 test subjects the CI width is
the honest summary, and it is wide.
