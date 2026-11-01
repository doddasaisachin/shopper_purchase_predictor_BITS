# 🛒 Shopper Purchase Predictor

ML Assignment 2 — M.Tech (AIML), BITS Pilani WILP

- **GitHub repository:** `<ADD-YOUR-GITHUB-REPO-LINK>`
- **Live Streamlit app:** `<ADD-YOUR-STREAMLIT-APP-LINK>`

---

## a. Problem statement

E-commerce sites see thousands of browsing sessions a day, but only a small fraction end in a purchase.
The goal is to predict, from the clickstream and session attributes of a visit, **whether the session will
end in a purchase (`Revenue = TRUE`) or not**. This is a **binary classification** problem with a class
imbalance (≈15.5 % positive), so beyond accuracy we focus on AUC, F1 and MCC.

Five classifiers — Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes and an
RBF Support Vector Machine — are trained on the same data, compared on six metrics, and served through an
interactive Streamlit app that evaluates the saved models on an uploaded test CSV.

## b. Dataset description

| Item | Value |
|---|---|
| Name | Online Shoppers Purchasing Intention Dataset |
| Source | UCI Machine Learning Repository — <https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset> |
| Reference | Sakar, C.O., Polat, S.O., Katircioglu, M. et al. *Neural Computing & Applications* (2018) |
| Instances | 12,330 sessions (12,205 after removing 125 exact duplicates) |
| Features | 17 (10 numeric + 7 categorical) |
| Target | `Revenue` — TRUE (purchase) / FALSE (no purchase) |
| Class balance | 84.4 % FALSE / 15.6 % TRUE |
| Missing values | None |

**Features**

| Feature | Type | Meaning |
|---|---|---|
| `Administrative`, `Informational`, `ProductRelated` | numeric | Number of pages of each type visited in the session |
| `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration` | numeric | Seconds spent on each page type |
| `BounceRates` | numeric | Avg. bounce rate (Google Analytics) of pages visited |
| `ExitRates` | numeric | Avg. exit rate of pages visited |
| `PageValues` | numeric | Avg. page value of pages visited before a transaction |
| `SpecialDay` | numeric | Closeness of visit date to a special day (e.g. Valentine's) |
| `Month` | categorical | Month of the visit |
| `OperatingSystems`, `Browser`, `Region`, `TrafficType` | categorical (integer codes) | Visitor technology / geography / traffic source |
| `VisitorType` | categorical | Returning, New or Other visitor |
| `Weekend` | categorical (bool) | Whether the visit was on a weekend |

**Pre-processing** (inside each saved scikit-learn `Pipeline`, so the app needs no separate step)

- Duplicates dropped; target mapped TRUE→1 / FALSE→0.
- Numeric features → `StandardScaler`.
- Categorical features (including integer-coded OS/Browser/Region/TrafficType, which have no ordinal
  meaning) → `OneHotEncoder(handle_unknown="ignore")`.
- Stratified 80 / 20 train-test split (`random_state=42`): **9,764 train / 2,441 test** rows.
  The test split is saved as `test_data.csv`.
- Hyper-parameters tuned with 5-fold stratified `GridSearchCV` (scoring = F1) on the **training set only**.

## c. GitHub repository

```
shopper-purchase-predictor/
├── app.py                         # Streamlit app (loads saved models, no retraining)
├── requirements.txt
├── README.md
├── test_data.csv                  # held-out 20 % test split (features + Revenue)
├── .streamlit/config.toml         # app theme
├── data/
│   └── online_shoppers_intention.csv   # raw UCI dataset
└── model/
    ├── train_models.py            # training + evaluation script
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── metrics.csv                # comparison table below
    └── model_meta.json            # feature lists, tuned hyper-parameters
```

Reproduce:

```bash
pip install -r requirements.txt
python model/train_models.py     # retrains all 5 models, rewrites model/ and test_data.csv
streamlit run app.py
```

## d. Models used

| Model | Key settings (tuned values in bold) |
|---|---|
| Logistic Regression | L2, `class_weight="balanced"`, **C = 0.1** |
| Decision Tree | `class_weight="balanced"`, **max_depth = 4**, **min_samples_leaf = 20** |
| KNN | Euclidean, **k = 5**, **weights = uniform** |
| Naive Bayes | Gaussian, **var_smoothing = 0.1** |
| Support Vector Machine | RBF kernel, `class_weight="balanced"`, `probability=True`, **C = 1.0** |

### Comparison table (held-out test set, 2,441 rows, threshold = 0.5 on P(purchase))

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8488 | 0.9108 | 0.5110 | 0.7932 | 0.6215 | 0.5527 |
| Decision Tree | 0.8583 | **0.9259** | 0.5291 | **0.8560** | **0.6540** | **0.5972** |
| KNN | 0.8734 | 0.7958 | 0.6622 | 0.3901 | 0.4909 | 0.4435 |
| Naive Bayes | 0.8161 | 0.8418 | 0.4419 | 0.6675 | 0.5318 | 0.4371 |
| Support Vector Machine | **0.8845** | 0.9095 | **0.6656** | 0.5262 | 0.5877 | 0.5265 |

> Baseline: always predicting "no purchase" already gives 84.4 % accuracy, so accuracy alone is a poor
> guide here — MCC and F1 are the more honest summary metrics.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong, stable baseline (AUC 0.911). `class_weight="balanced"` pushes recall to 0.79 at the cost of precision (0.51) — it catches most buyers but flags many non-buyers. Grid search preferred strong regularisation (C = 0.1), suggesting the signal is concentrated in a few features (mainly `PageValues`, `ExitRates`, `Month`). Being linear, it cannot model interactions such as "high PageValues **and** November". |
| Decision Tree | Best overall: highest AUC (0.926), recall (0.856), F1 (0.654) and MCC (0.597). A shallow depth-4 tree was optimal — deeper trees overfit. It captures the sharp threshold behaviour of `PageValues` (83 % of the tree's feature importance) and its interaction with `Month = Nov` (8 %) directly, and remains fully interpretable. |
| KNN | Accuracy (0.873) looks good but is misleading: recall is only 0.39 and AUC (0.796) is the lowest. With 74 dimensions after one-hot encoding, distances become less informative, and since KNN has no class weighting the majority "no purchase" class dominates every neighbourhood. It is also the heaviest model to store (keeps all training rows) and the slowest at prediction time. |
| Naive Bayes | Weakest performer (accuracy 0.816, MCC 0.437). Its feature-independence assumption is clearly violated (e.g. `BounceRates` vs `ExitRates` correlation = 0.91, page counts vs durations are highly correlated) and the numeric features are heavily right-skewed rather than Gaussian. A large `var_smoothing` (0.1) was needed to stabilise it. Fast and simple, but not competitive here. |
| Support Vector Machine | Highest accuracy (0.885) and precision (0.666) with AUC 0.910 — ranking quality on par with Logistic Regression. At the default 0.5 threshold on its Platt-calibrated probabilities it is conservative (recall 0.53); lowering the threshold to 0.30 in the app raises F1 to ≈ 0.66 and MCC to ≈ 0.59, close to the Decision Tree. Training is the slowest of the five. |
| **Overall winner for this dataset** | **Decision Tree** — best MCC, F1, AUC and recall, while being the most interpretable and cheapest to serve. SVM is the runner-up and preferable if precision (fewer false "will buy" alerts) matters more than recall. |

## e. Streamlit app features

- **CSV upload** of test data (or one-click use / download of the bundled `test_data.csv`).
- **Model selection dropdown** for the five classifiers, plus an adjustable decision-threshold slider.
- **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1, MCC.
- **Confusion matrix, ROC curve and classification report** for the selected model.
- **Compare-all tab**: all five models evaluated live on the uploaded data with a metric bar chart.
- **Predictions tab**: row-level probabilities, misclassified-only filter and CSV download.

The uploaded CSV must contain all 17 feature columns and the `Revenue` column (TRUE/FALSE or 1/0).
Models are loaded from `model/*.joblib`; nothing is retrained in the app.
