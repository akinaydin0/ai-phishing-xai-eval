# HPO + Classical/Ensemble Grid + Dense Embeddings — Summary

Optuna HPO (val ROC-AUC objective, tie-break val F1; test untouched during tuning) for 7 families × 3 corpora, plus dense-embedding baselines and champion cross-corpus transfer. Linear families use 100k TF-IDF; tree/boosting families **tune** max_features ∈ {10k, 20k, 50k}. XGB/LGBM fix n_estimators=800 with early_stopping_rounds=50 on val AUC. XGBoost search ran on GPU (RTX 3090); every finalist retrain + single test eval ran on CPU (seed 42, deterministic). Full log: `results/results.csv`.

## (a) Best hyperparameters per (family, corpus)

| Corpus | Family | val AUC | search | max_feat | best params |
|---|---|--:|:--:|--:|---|
| d1_sb | LogisticRegression | 0.9997 | cpu | 100k | C=94.16472976629126, ngram=1_2, min_df=5, sublinear=False |
| d1_sb | MultinomialNB | 0.9994 | cpu | 100k | alpha=0.0010031790577147317, ngram=1_1, min_df=1, sublinear=False |
| d1_sb | LinearSVC | 0.9997 | cpu | 100k | C=0.3332268568195022 |
| d1_sb | DecisionTree | 0.9846 | cpu | 50000 | max_depth=51, min_samples_leaf=32 |
| d1_sb | RandomForest | 0.9988 | cpu | 20000 | n_estimators=532, max_depth=None, max_features=0.3 |
| d1_sb | XGBoost | 0.9993 | cuda | 20000 | learning_rate=0.06181931957263936, max_depth=6, subsample=0.999765397635074, colsample_bytree=0.5984296098621205 best_iter=797 |
| d1_sb | LightGBM | 0.9996 | cpu | 20000 | learning_rate=0.08319577644332436, num_leaves=32, feature_fraction=0.9847685553939328, bagging_fraction=0.9729161367647149 best_iter=314 |
| d2 | LogisticRegression | 0.9991 | cpu | 100k | C=24.682706336199583, ngram=1_2, min_df=5, sublinear=False |
| d2 | MultinomialNB | 0.9990 | cpu | 100k | alpha=0.0011065136021090993, ngram=1_2, min_df=5, sublinear=False |
| d2 | LinearSVC | 0.9991 | cpu | 100k | C=6.224909951304237 |
| d2 | DecisionTree | 0.9763 | cpu | 20000 | max_depth=50, min_samples_leaf=38 |
| d2 | RandomForest | 0.9974 | cpu | 50000 | n_estimators=734, max_depth=None, max_features=log2 |
| d2 | XGBoost | 0.9983 | cuda | 10000 | learning_rate=0.1452181485888502, max_depth=5, subsample=0.9621264648206656, colsample_bytree=0.6420518258597355 best_iter=790 |
| d2 | LightGBM | 0.9985 | cpu | 20000 | learning_rate=0.025484045425698022, num_leaves=56, feature_fraction=0.7145622178248661, bagging_fraction=0.9362148766913247 best_iter=639 |
| d3_en | LogisticRegression | 0.9975 | cpu | 100k | C=89.3855774195032, ngram=1_2, min_df=5, sublinear=False |
| d3_en | MultinomialNB | 0.9867 | cpu | 100k | alpha=0.001003272149149236, ngram=1_2, min_df=1, sublinear=True |
| d3_en | LinearSVC | 0.9989 | cpu | 100k | C=9.587479255847708 |
| d3_en | DecisionTree | 0.9354 | cpu | 20000 | max_depth=51, min_samples_leaf=22 |
| d3_en | RandomForest | 0.9894 | cpu | 20000 | n_estimators=696, max_depth=70, max_features=0.3 |
| d3_en | XGBoost | 0.9959 | cuda | 10000 | learning_rate=0.07312008956366879, max_depth=9, subsample=0.999765397635074, colsample_bytree=0.8409983993043059 best_iter=784 |
| d3_en | LightGBM | 0.9972 | cpu | 20000 | learning_rate=0.08496761732351853, num_leaves=33, feature_fraction=0.9859113542175745, bagging_fraction=0.9787847505560596 best_iter=340 |

## (b) Tuned vs. untuned-default (test set) — tuning delta

| Corpus | Family | default F1 | tuned F1 | ΔF1 | default AUC | tuned AUC | ΔAUC |
|---|---|--:|--:|--:|--:|--:|--:|
| d1_sb | LogisticRegression | 0.9906 | 0.9939 | +0.0033 | 0.9995 | 0.9997 | +0.0002 |
| d1_sb | MultinomialNB | 0.9779 | 0.9870 | +0.0091 | 0.9985 | 0.9994 | +0.0009 |
| d1_sb | LinearSVC | 0.9955 | 0.9949 | -0.0006 | 0.9998 | 0.9998 | +0.0000 |
| d1_sb | DecisionTree | 0.9681 | 0.9563 | -0.0118 | 0.9649 | 0.9864 | +0.0215 |
| d1_sb | RandomForest | 0.9861 | 0.9875 | +0.0014 | 0.9989 | 0.9993 | +0.0004 |
| d1_sb | XGBoost | 0.9883 | 0.9906 | +0.0023 | 0.9993 | 0.9995 | +0.0002 |
| d1_sb | LightGBM | 0.9887 | 0.9917 | +0.0030 | 0.9994 | 0.9996 | +0.0002 |
| d2 | LogisticRegression | 0.9876 | 0.9916 | +0.0040 | 0.9992 | 0.9995 | +0.0003 |
| d2 | MultinomialNB | 0.9860 | 0.9903 | +0.0043 | 0.9989 | 0.9996 | +0.0007 |
| d2 | LinearSVC | 0.9933 | 0.9925 | -0.0008 | 0.9995 | 0.9994 | -0.0001 |
| d2 | DecisionTree | 0.9387 | 0.9267 | -0.0120 | 0.9416 | 0.9760 | +0.0344 |
| d2 | RandomForest | 0.9847 | 0.9846 | -0.0001 | 0.9984 | 0.9987 | +0.0003 |
| d2 | XGBoost | 0.9778 | 0.9833 | +0.0055 | 0.9983 | 0.9989 | +0.0006 |
| d2 | LightGBM | 0.9837 | 0.9858 | +0.0021 | 0.9988 | 0.9992 | +0.0004 |
| d3_en | LogisticRegression | 0.9751 | 0.9856 | +0.0105 | 0.9949 | 0.9982 | +0.0033 |
| d3_en | MultinomialNB | 0.9442 | 0.9621 | +0.0179 | 0.9720 | 0.9870 | +0.0150 |
| d3_en | LinearSVC | 0.9895 | 0.9908 | +0.0013 | 0.9990 | 0.9993 | +0.0003 |
| d3_en | DecisionTree | 0.9122 | 0.9071 | -0.0051 | 0.8702 | 0.9232 | +0.0530 |
| d3_en | RandomForest | 0.9526 | 0.9592 | +0.0066 | 0.9874 | 0.9901 | +0.0027 |
| d3_en | XGBoost | 0.9733 | 0.9751 | +0.0018 | 0.9951 | 0.9957 | +0.0006 |
| d3_en | LightGBM | 0.9736 | 0.9818 | +0.0082 | 0.9956 | 0.9980 | +0.0024 |

## (c) In-corpus leaderboard (tuned models + dense), by test ROC-AUC

**d1_sb**

| Rank | Model | Features | F1 | ROC-AUC |
|--:|---|---|--:|--:|
| 1 | LinearSVC | tfidf | 0.9949 | 0.9998 |
| 2 | LogisticRegression | tfidf | 0.9939 | 0.9997 |
| 3 | LightGBM | tfidf | 0.9917 | 0.9996 |
| 4 | XGBoost | tfidf | 0.9906 | 0.9995 |
| 5 | MultinomialNB | tfidf | 0.9870 | 0.9994 |
| 6 | RandomForest | tfidf | 0.9875 | 0.9993 |
| 7 | MLP | w2v | 0.9907 | 0.9989 |
| 8 | MLP | glove | 0.9875 | 0.9974 |
| 9 | LogisticRegression | w2v | 0.9805 | 0.9966 |
| 10 | LogisticRegression | glove | 0.9659 | 0.9909 |
| 11 | DecisionTree | tfidf | 0.9563 | 0.9864 |

**d2**

| Rank | Model | Features | F1 | ROC-AUC |
|--:|---|---|--:|--:|
| 1 | MultinomialNB | tfidf | 0.9903 | 0.9996 |
| 2 | LogisticRegression | tfidf | 0.9916 | 0.9995 |
| 3 | LinearSVC | tfidf | 0.9925 | 0.9994 |
| 4 | LightGBM | tfidf | 0.9858 | 0.9992 |
| 5 | XGBoost | tfidf | 0.9833 | 0.9989 |
| 6 | RandomForest | tfidf | 0.9846 | 0.9987 |
| 7 | MLP | w2v | 0.9839 | 0.9975 |
| 8 | MLP | glove | 0.9796 | 0.9960 |
| 9 | LogisticRegression | w2v | 0.9748 | 0.9957 |
| 10 | LogisticRegression | glove | 0.9625 | 0.9902 |
| 11 | DecisionTree | tfidf | 0.9267 | 0.9760 |

**d3_en**

| Rank | Model | Features | F1 | ROC-AUC |
|--:|---|---|--:|--:|
| 1 | LinearSVC | tfidf | 0.9908 | 0.9993 |
| 2 | LogisticRegression | tfidf | 0.9856 | 0.9982 |
| 3 | LightGBM | tfidf | 0.9818 | 0.9980 |
| 4 | MLP | glove | 0.9780 | 0.9961 |
| 5 | XGBoost | tfidf | 0.9751 | 0.9957 |
| 6 | MLP | w2v | 0.9766 | 0.9942 |
| 7 | RandomForest | tfidf | 0.9592 | 0.9901 |
| 8 | MultinomialNB | tfidf | 0.9621 | 0.9870 |
| 9 | LogisticRegression | w2v | 0.9579 | 0.9853 |
| 10 | LogisticRegression | glove | 0.9483 | 0.9760 |
| 11 | DecisionTree | tfidf | 0.9071 | 0.9232 |

## (d) Champion cross-corpus (train → test)

| Group | train → test | Model | F1 | ROC-AUC |
|---|---|---|--:|--:|
| tfidf | d1_sb → d2 | LogisticRegression | 0.8245 | 0.9091 |
| tfidf | d1_sb → d3_en | LogisticRegression | 0.8310 | 0.8965 |
| ensemble | d1_sb → d2 | LightGBM | 0.7991 | 0.8987 |
| ensemble | d1_sb → d3_en | LightGBM | 0.8277 | 0.8688 |
| dense | d1_sb → d2 | MLP | 0.8170 | 0.8910 |
| dense | d1_sb → d3_en | MLP | 0.7402 | 0.8724 |
| tfidf | d2 → d1_sb | LogisticRegression | 0.7936 | 0.9080 |
| tfidf | d2 → d3_en | LogisticRegression | 0.8430 | 0.8429 |
| ensemble | d2 → d1_sb | LightGBM | 0.7734 | 0.8752 |
| ensemble | d2 → d3_en | LightGBM | 0.8179 | 0.7948 |
| dense | d2 → d1_sb | MLP | 0.7076 | 0.8357 |
| dense | d2 → d3_en | MLP | 0.8300 | 0.8102 |
| tfidf | d3_en → d1_sb | LinearSVC | 0.7139 | 0.8783 |
| tfidf | d3_en → d2 | LinearSVC | 0.6874 | 0.7523 |
| ensemble | d3_en → d1_sb | LightGBM | 0.7430 | 0.8888 |
| ensemble | d3_en → d2 | LightGBM | 0.6551 | 0.7052 |
| dense | d3_en → d1_sb | MLP | 0.6529 | 0.7914 |
| dense | d3_en → d2 | MLP | 0.6379 | 0.6751 |

Champions per source (by val AUC): tfidf|d1_sb=lr; ensemble|d1_sb=lgbm; dense|d1_sb=w2v+MLP; tfidf|d2=lr; ensemble|d2=lgbm; dense|d2=w2v+MLP; tfidf|d3_en=svc; ensemble|d3_en=lgbm; dense|d3_en=glove+MLP

## (e) Computational cost (per-fit calibration, results/probe_timings.json)

| Model | CPU single-fit @100k (s) | GPU (XGB @20k) | study runtime (sum 3 corpora, s) |
|---|--:|--:|--:|
| LogisticRegression | 0.7 | — | 182 |
| MultinomialNB | 0.0 | — | 51 |
| LinearSVC | 0.5 | — | 56 |
| DecisionTree | 19.7 | — | 716 |
| RandomForest | 7.9 | — | 9040 |
| XGBoost | 605.8 | 3.4s (105.9× vs 360.2s CPU) | 4655 |
| LightGBM | 242.8 | — | 11774 |

*XGBoost is ~106× faster on the RTX 3090 GPU at 20k features (3.4 s vs 360 s CPU); without the GPU the XGB search alone would exceed 10 h. This feeds the dissertation's accuracy-vs-computational-cost analysis (to be extended with transformer timings).*

- **XGBoost** tuned TF-IDF max_features (d1_sb / d2 / d3_en): 20000 / 10000 / 10000 — capped to {10k, 20k} by the 24 GB GPU ceiling at 50k×depth-10 (see notes/decisions.md).
- **LightGBM** tuned TF-IDF max_features (d1_sb / d2 / d3_en): 20000 / 20000 / 20000 — full {10k, 20k, 50k} on CPU, the empirical 50k check.

## Tuned TF-IDF max_features winner per tree/boosting family

| Corpus | DecisionTree | RandomForest | XGBoost | LightGBM |
|---|--:|--:|--:|--:|
| d1_sb | 50000 | 20000 | 20000 | 20000 |
| d2 | 20000 | 50000 | 10000 | 20000 |
| d3_en | 20000 | 20000 | 10000 | 20000 |

*Fairness note: linear models use 100k features, tree/boosting models use their tuned max_features — the deployment-relevant comparison is each family at its best configuration, not a single shared representation.*

## Observations

1. **Tuned linear TF-IDF is the in-corpus champion; the ensembles and dense models do not beat it.** The top model by test ROC-AUC is a linear TF-IDF model on every corpus — LinearSVC (0.9998) on d1_sb, MultinomialNB (0.9996) on d2, LinearSVC (0.9993) on d3_en — with LightGBM/XGBoost a hair behind and the best dense model (GloVe-MLP) only reaching the top-5 on the AI corpus. On high-dimensional sparse TF-IDF, linear models remain both strongest and cheapest — a standard result reaffirmed here after full HPO.

2. **HPO gains are small and near-ceiling — largest where the default is weak.** Tuning lifts test F1 by roughly +0.002 to +0.018, with the biggest gains on the weakest defaults (MultinomialNB ΔF1 +0.0091/+0.0043/+0.0179). LinearSVC is already at its optimum by default (ΔF1 -0.0006/-0.0008/+0.0013). These corpora sit near the performance ceiling (in-corpus AUC ≥ 0.99), so HPO buys little in-corpus; its value is expected to show on harder data and cross-corpus.

3. **Tuning for AUC can trade away threshold-F1 — DecisionTree is the tell.** DT is the one family where tuning *lowers* F1 (ΔF1 -0.0118/-0.0120/-0.0051) while *raising* AUC sharply (ΔAUC +0.0215/+0.0344/+0.0530). The val-AUC objective selects trees that rank better but whose fixed-0.5-threshold F1 falls — reinforcing the baselines' finding that the operating threshold must be set separately from the ranking model.

4. **Tree/boosting models prefer ~20k features, validating the tuned-dimension decision.** Given {10k, 20k, 50k} (XGB capped to {10k, 20k}), the winners cluster at 20k — LightGBM chose 20k on all three corpora (20000/20000/20000) — and never approach the linear models' 100k. Histogram methods genuinely cannot exploit ultra-high-dimensional sparse text, so tuning max_features per family (rather than sharing the 100k default) was the correct, fairer design; each family is compared at its own best representation.

5. **Champion cross-corpus transfer echoes the baselines: strong AUC but asymmetric, AI→real weakest.** Champion cross-corpus ROC-AUC spans 0.675–0.909. Real→AI transfer is strong (mean AUC to d3_en 0.848) while AI→real is weaker (mean AUC from d3_en 0.782) — the same asymmetry the TF-IDF LR baselines showed, now confirmed for the tuned champions across model families. The diagonal (in-corpus) stays ≥0.99; the off-diagonal degradation is the dataset-shift signal the framework is built to surface.
