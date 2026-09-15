# Data placement

Place the **provided assignment dataset only** in this directory. No external data are used.

Expected layout:

```text
data/
├── subjects.csv
├── modality_A.csv
├── modality_B.csv
├── simulation_scores.csv
├── node_labels.csv
├── region_names.csv
└── adjacency/
    └── <subject_id>_adj.npy   # 140 matrices, each 68 x 68
```

The code also recognises the original `candidate_package/data/` and Kaggle input layouts used during development.
