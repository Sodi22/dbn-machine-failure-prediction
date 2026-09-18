# DBN Machine-Failure Prediction — All Variants and Their Scores

Summary of the five notebooks in this repo. All of them answer the same question

> **P(State₍t+1₎ = Failure | State₍t₎, alarm features at t)**

and differ only in **which alarms they watch**, **how the next state is wired to those
alarms**, **how the CPTs are estimated**, and **what they are evaluated on**.

> **Scores are not filled in yet.** None of the notebooks has stored outputs, and
> `dataset_exercise.csv` is not in the repo (it is gitignored), so the numbers below
> cannot be read off without re-running. See
> [Filling in the score table](#filling-in-the-score-table) at the end.

---

## 1. The shared pipeline

Every variant does the same four preprocessing steps:

1. **Duration** — `duration_seconds = end_alarm - start_alarm` per alarm record.
2. **Discretisation** — raw numbers → categories, because BNs need discrete values:

   | Count | Category | | Duration (s) | Category |
   |-------|----------|-|--------------|----------|
   | 0     | None     | | 0            | None     |
   | 1–2   | Low      | | 1–30         | Short    |
   | 3–5   | Medium   | | 31–300       | Medium   |
   | >5    | High     | | >300         | Long     |

3. **One row per `time_window`** — majority `machine_state` + each alarm's count and
   total duration, discretised.
4. **Transitions** — `shift(-1)` pairs window *t* with *t+1*, keeping only pairs whose
   `time_window` IDs are exactly 1 apart (the column has gaps).

What differs from there is the model.

---

## 2. The five variants

| # | Notebook | Alarms watched | Features | Graph into `State_t+1` | Estimator | Inference | Evaluated on |
|---|----------|----------------|----------|------------------------|-----------|-----------|--------------|
| 1 | `dbn_failure_prediction.ipynb` | top 3 most frequent | 6 | **full joint** — `State_t` + all 6 features are parents | Bayesian, **BDeu** (ESS = 5) | `VariableElimination` | held-out **last 25 %** |
| 2 | `dbn_alarm_data_prediction.ipynb` | top 3 most frequent | 6 | **full joint**, real `DynamicBayesianNetwork` | **MLE** via `dbn.fit()` | `DBNInference` | **all** transitions (in-sample) |
| 3 | `dbn_failure_prediction_all_alarms.ipynb` | **all 94** | 188 | **Naive-Bayes** — `State_t → State_t+1 → each feature` | Bayesian, **BDeu** (ESS = 5) | `VariableElimination` | held-out **last 25 %** |
| 4 | `dbn_alarm_data_prediction_all_alarms.ipynb` | **all 94** | 188 | **Naive-Bayes** (same as #3) | **MLE** | `VariableElimination` | **all** transitions (in-sample) |
| 5 | `dbn_fit_all_alarms_aggregated.ipynb` | **all 94**, aggregated | 4 | **full joint**, real `DynamicBayesianNetwork` | **MLE** via `dbn.fit()` | `DBNInference` | **all** transitions (in-sample) |

### 1 — BDeu, top 3 alarms (reference model)

`State_t` plus the six alarm features are all parents of `State_t+1`, so the CPT keeps
the **full interaction** between alarms (2 × 4⁶ = 8 192 parent configurations). Trained
with a **BDeu prior**, so alarm combinations never seen in training get a small non-zero
probability instead of zero. Chronological 75/25 split; reports accuracy, precision,
recall, F1, confusion matrix and a **persistence baseline** (`next = current`).

### 2 — MLE, top 3 alarms (textbook DBN)

The closest thing to a textbook DBN: a real `DynamicBayesianNetwork`, `dbn.fit(estimator='MLE')`,
`DBNInference`. Self-transition edges `(feature, 0) → (feature, 1)` are added only because
`DBN.fit` requires every variable to exist at both slices. Two consequences to keep in mind
when reading its score:

- **No smoothing** — unseen parent configurations get probability 0, and the eval loop
  falls back to *persistence* whenever a query raises.
- **In-sample** — it scores on `df_transitions`, the same rows it was trained on, and
  reports accuracy + confusion only (no F1, no baseline).

### 3 — BDeu, all 94 alarms (Naive-Bayes)

One `State` node cannot depend jointly on 188 features — the table would need ~10¹¹³
entries — so the alarms are made **conditionally independent given the state**:
`State_t → State_t+1`, and `State_t+1 → each alarm feature`. Each alarm then gets its own
small CPT. Otherwise identical to #1 (BDeu, 75/25, full metric set), which makes #1 vs #3
a **clean comparison of feature-set size** on the same protocol.

### 4 — MLE, all 94 alarms (Naive-Bayes)

Same Naive-Bayes structure as #3 but with **MLE** instead of BDeu, trained and evaluated
on all transitions. So #3 vs #4 isolates the **prior**, and #2 vs #4 isolates the
**feature set** — each within its own protocol.

### 5 — Pure `dbn.fit()`, all 94 alarms aggregated

Demonstrates that the *pure* DBN path can cover all 94 alarms without any Naive-Bayes
flattening, by summarising each window into the exercise's two attributes, two ways each:

- **Count** — `total_alarms` (all firings) and `num_active` (distinct alarms)
- **Duration** — `total_dur` (combined) and `max_dur` (longest single alarm)

Each is discretised into None / Low / Medium / High, with the cut-points at the **tertiles
of the non-zero values** so the many zeros don't skew the bins. A real `dbn.fit()` +
`DBNInference` model, evaluated in-sample.

---

## 3. Score table

Fill from the evaluation cell of each notebook. Positive class = **Failure**.

| # | Variant | Accuracy | Precision | Recall | F1 | TP / FP / FN / TN | Persistence baseline |
|---|---------|----------|-----------|--------|----|-------------------|----------------------|
| 1 | BDeu, top 3 (held-out 25 %) | – | – | – | – | – | – |
| 2 | MLE, top 3, pure DBN (in-sample) | – | n/a¹ | n/a¹ | n/a¹ | – | n/a¹ |
| 3 | BDeu, all 94, Naive-Bayes (held-out 25 %) | – | – | – | – | – | – |
| 4 | MLE, all 94, Naive-Bayes (in-sample) | – | – | – | – | – | n/a² |
| 5 | Pure `dbn.fit()`, 94 aggregated (in-sample) | – | – | – | – | – | n/a² |

¹ Notebook 2 prints accuracy and the confusion matrix only — precision/recall/F1 can be
derived from TP/FP/FN/TN if needed.
² Notebooks 4 and 5 don't print the persistence baseline.

**Read the table with two cautions:**

- **Only #1 and #3 are directly comparable.** They are the only two scored on data the
  model never saw. #2, #4 and #5 score on their own training rows, which inflates every
  number — an in-sample accuracy above a held-out one is not evidence of a better model.
- **Accuracy alone is misleading.** `Failure` is the minority class, so a model that
  predicts `Running` everywhere already scores high accuracy. **F1 on Failure, compared
  against the persistence baseline**, is the metric that says whether the alarm features
  add anything beyond "assume nothing changes".

---

## 4. Why the variants score the way they do

The four design choices that move the numbers, and the direction each one pushes:

### Full joint CPT vs Naive-Bayes (#1 vs #3, #2 vs #4)

The joint CPT over 6 features can express *combinations* — "A1 long **and** A2 frequent"
— which is exactly the kind of pattern that precedes a failure. Naive-Bayes cannot: it
multiplies each feature's evidence independently, so interactions are lost. The joint
table costs 8 192 parent configurations, but with only 6 features and BDeu smoothing
filling the gaps, that is affordable. This is the main reason the top-3 models are the
stronger ones.

### 3 alarms vs 94 (#1 vs #3)

Most of the 94 alarms are **almost always idle**, so their features sit at
`None` / `None` in nearly every window. Each such feature contributes a likelihood ratio
of roughly 1 — no information — but there are **188 of them**, and under the Naive-Bayes
product their accumulated noise swamps the handful of genuinely informative alarms and
the `State_t` prior. The predictable symptom is a posterior pulled toward the majority
class, i.e. **collapsing recall on Failure**: the model stops raising the alarm. Adding
features here *removes* signal rather than adding it, which is why the top-3 notebooks
are the primary models.

### BDeu vs MLE (#1 vs #2, #3 vs #4)

With 8 192 parent configurations and only a few hundred transitions, the training data
covers a small fraction of the table. **MLE assigns probability 0 to everything unseen**,
so a single unseen alarm combination at prediction time makes the query degenerate or
raise — and in notebook 2 that falls back to persistence, meaning part of its score is
really the baseline's score, not the model's. **BDeu (ESS = 5)** instead spreads a small
amount of prior mass over unseen cells, which keeps inference well-defined and makes the
held-out numbers trustworthy. Expect BDeu to be the more honest and the more robust of
the two; MLE's advantage in raw in-sample accuracy is largely memorisation plus baseline
fallback.

### Aggregation (#5)

Collapsing 94 alarms into four totals keeps the model a genuine joint DBN and keeps the
CPT small, but it **discards which alarm fired**. A failure driven by one specific alarm
becomes indistinguishable from general busyness, so its signal is averaged away — the
expected symptom is again **low recall**. `max_dur` (longest single alarm) is the feature
that retains the most of it, since windows containing one very long alarm fail
disproportionately often. This notebook is best read as a **feasibility demonstration**
of the pure `dbn.fit()` path over all 94 alarms, not as the best predictor.

### Expected ranking

On held-out F1 for the Failure class:

**#1 (BDeu, top 3) > #3 (BDeu, all 94) > #5 (aggregated)**, with the MLE variants (#2, #4)
posting higher in-sample numbers that don't transfer. Ordering #2 and #4 against the rest
requires re-running them on the same 75/25 split — see below.

---

## 5. Filling in the score table

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
# place dataset_exercise.csv (semicolon-separated) in this folder
jupyter lab                     # Kernel -> Restart & Run All
```

Then copy the printed block from each notebook's evaluation cell into the table:

| # | Notebook | Cell to read |
|---|----------|--------------|
| 1 | `dbn_failure_prediction.ipynb` | the `EVALUATION (positive class = Failure)` block |
| 2 | `dbn_alarm_data_prediction.ipynb` | the `Total Evaluated / TP / FP / TN / FN / Accuracy` block |
| 3 | `dbn_failure_prediction_all_alarms.ipynb` | the `EVALUATION (positive class = Failure)` block |
| 4 | `dbn_alarm_data_prediction_all_alarms.ipynb` | the `Total Evaluated … F1 Score` block |
| 5 | `dbn_fit_all_alarms_aggregated.ipynb` | the `PURE DBN (aggregated, all 94 alarms)` block |

**To make all five strictly comparable**, two changes are needed in #2, #4 and #5:

1. Apply the same chronological 75/25 split used in #1 and #3 (`split = int(len(...) * 0.75)`)
   and evaluate on the test slice only.
2. Print precision / recall / F1 and the persistence baseline
   (`(test['State_t'] == test['State_t1']).mean()`) alongside accuracy.

Until then the in-sample rows should be read as *fit quality*, not predictive performance.
