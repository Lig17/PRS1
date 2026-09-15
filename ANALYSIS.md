# ANALYSIS.md

**Candidate:** Lighittha P R  
**Email:** ligpersonal@gmail.com


## Required question

> **My graph model beats a non-graph baseline. How do I determine whether graph structure is genuinely contributing rather than the difference being noise or an artefact?**

I would not accept a single GNN-minus-baseline score as evidence. A small gain can come from seed variance, a stronger function class, a favorable split, leakage, regularisation, or a few influential subjects. I therefore test the graph claim through matched controls, repeated seeds, subject-level uncertainty, capacity sensitivity and error analysis.

### 1. Identity adjacency: does message passing add anything beyond the architecture?

I replace the graph operator with identity while keeping the same model family, parameterisation, optimiser, training budget and early-stopping logic. This removes neighbour exchange while preserving the neural architecture. On validation across 10 paired seeds, real adjacency gives **0.5413±0.0090 AUPRC** and identity gives **0.4584±0.0180**. The paired real-minus-identity delta is +0.0829 (SD 0.0226) and has the same sign on 10/10 seeds. On test, the ensemble difference is **+0.0995**, with 95% subject-bootstrap CI **[+0.0413, +0.1614]**.

This rejects the explanation that the gain is simply “a neural network is better than the non-graph baseline.”

### 2. Shuffled adjacency: would any smoothing graph work?

I permute node order in the adjacency while leaving feature/label rows fixed. This preserves graph size, density, degree distribution and edge-weight distribution but destroys the correspondence between a node's features and its true neighbours. A generic smoothing/regularisation benefit should survive this control; a topology-specific benefit should not.

Across 10 validation seeds, shuffled adjacency gives **0.4674±0.0121 AUPRC**, versus 0.5413 for real. The paired delta is +0.0739 (SD 0.0161), positive on 10/10 seeds. On test, real is **0.5818** and shuffled is **0.4985**, a difference of **+0.0833 [ +0.0301, +0.1410 ]** by subject bootstrap.

This is my strongest evidence that the **correct node-topology correspondence**, rather than generic message passing, is contributing.

### 3. Weight-shuffled adjacency: is the signal topology or exact edge strength?

I keep the binary topology fixed and permute the non-zero edge weights. The test AUPRC becomes **0.5707**, only 0.0111 below real, with CI **[-0.0149, +0.0282]** for AUPRC. I therefore do not claim that precise edge magnitudes are essential. The dominant effect appears to be **which nodes are connected**, not the exact weight assigned to every edge.

### 4. Repeated seeds and paired variability

I use 10 seeds for each required graph condition. Pairing comparisons by seed keeps initialisation and stochastic training effects matched, so the adjacency condition is the main change. Real-minus-identity and real-minus-shuffled are positive on **10/10 seeds**. This makes a one-seed explanation implausible.

### 5. Subject-level confidence intervals, not node-IID intervals

The 68 nodes within a subject share the same graph and subject-level context, so treating 1,904 test nodes as independent would overstate certainty. I bootstrap **subjects**, drawing 28 subjects with replacement and retaining all of each selected subject's nodes, for 2,000 replicates. Both required graph contrasts have 95% CIs excluding zero:

- real vs identity: **+0.0995 [ +0.0413, +0.1614 ] AUPRC**;
- real vs shuffled: **+0.0833 [ +0.0301, +0.1410 ] AUPRC**.

I also report graph versus both non-graph families. Relative to the validation-selected HGB baseline, the AUPRC gain is +0.0906 [0.0436, 0.1321]. Relative to logistic regression, which happens to be stronger on test, it is +0.0561 [-0.0030, 0.1159]. I report both rather than selecting the easier comparator after seeing test results.

### 6. Baseline fairness

My non-graph models receive the same node feature matrix, including graph-derived **local** statistics such as weighted degree and eigen-centrality. This matters because my training EDA shows those local graph statistics already separate abnormal and normal nodes. Therefore real-versus-identity/shuffled asks a narrower and fairer question: **does propagation over the actual graph add information beyond topology that can be summarised as row-wise features?**

### 7. Capacity and design sensitivity

A graph gap could be an artefact of one width/depth setting. I therefore rerun real, shuffled and identity at three capacities: h=16/L=1, h=64/L=2 and h=128/L=3. Real remains above both controls at every capacity; the smallest real-minus-identity margin is +0.0712 and the smallest real-minus-shuffled margin is +0.0621 AUPRC.

I also ablate self-loop choice, symmetric versus weighted-mean aggregation, and BCE versus focal loss. These checks show that the main real-vs-control ordering is not dependent on one fragile design decision. The custom backward pass is additionally verified by finite differences (maximum relative gradient error 1.3e-08).

### 8. Error analysis: which nodes and subjects does the graph fix?

At test top-k, real message passing recovers **22 positive nodes that identity misses** and loses 4 positives that identity gets. At the subject level, **13 subjects improve, 0 worsen, and 15 are unchanged**. Removing the two most-improved subjects still leaves a mean Dice improvement of +0.1107 versus +0.1409 overall, so the effect is not a two-subject anecdote.

I had a mechanistic hypothesis from training EDA: abnormal nodes are more strongly interconnected and have a higher weighted fraction of abnormal neighbours. I therefore compare neighbourhood properties of graph-fixed versus graph-harmed nodes. This specific mechanism test is **underpowered**: there are only 22 fixed and 4 harmed nodes, and the observed abnormal-neighbour fractions are 0.134 versus 0.130. I do not count this as positive evidence. It means I have strong evidence **that** topology helps, but not enough evidence to establish **why** at the individual-node mechanism level.

### 9. What result would make me conclude the graph does not help?

I would reject the graph-contribution claim if real and identity were indistinguishable, if real and shuffled had a CI crossing zero, if seed-paired differences frequently changed sign, if the effect vanished after removing one or two influential subjects, or if the ordering disappeared across reasonable model capacities.

The first four quantitative checks are not observed: real beats identity and shuffled with positive subject-level CIs, both paired deltas are positive on 10/10 seeds, the subject-level effect survives removing the strongest cases, and the ordering persists across three capacities. The node-level mechanism check remains unresolved and I report it as such.

## Conclusion

My conclusion is that **graph structure genuinely contributes to node localisation in this dataset**, with the strongest evidence coming from the matched real-vs-shuffled and real-vs-identity controls, repeated paired seeds, and subject-level bootstrap uncertainty. The additional weight-shuffle result narrows the claim: most of the benefit appears to come from the **correct topology**, while exact edge weights make a much smaller contribution. I do not claim causality, transfer to another parcellation, or a proven homophily mechanism.
