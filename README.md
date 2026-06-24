# Machine-Failure Prediction with Dynamic Bayesian Networks

Predict whether a machine will enter a **Failure** state in the next time window,
based on its current state and recent alarm activity:

> **P(State₍t+1₎ = Failure | State₍t₎, alarm features at t)**

Two notebooks implement this with [`pgmpy`](https://pgmpy.org/), using two different
training methods.

## The two methods

**`dbn_failure_prediction.ipynb` — BDeu-smoothed**
Declares the DBN structure, then learns the probabilities on an equivalent flat
Bayesian network using a **BDeu (Bayesian) prior**. Smoothing gives unseen alarm
combinations a small non-zero probability instead of zero, so inference stays robust.
Trained on the first 75% of the data and evaluated on the held-out last 25%.
Inference uses Variable Elimination.

**`dbn_alarm_data_prediction.ipynb` — MLE**
Uses `pgmpy`'s `DynamicBayesianNetwork` directly, with parameters learned by
**Maximum Likelihood Estimation** and inference via `DBNInference`. Simpler and
closer to the textbook DBN, but unseen alarm combinations get zero probability.

Both share the same pipeline: compute alarm durations → discretise counts and
durations into categories → pick the 3 most frequent alarms as features → build one
row per time window → form consecutive (t → t+1) transitions → train and predict.

## All-alarms variants

Two extra notebooks watch **all 94 alarms** instead of the 3 most frequent:

- **`dbn_failure_prediction_all_alarms.ipynb`** — BDeu
- **`dbn_alarm_data_prediction_all_alarms.ipynb`** — MLE

A single `State` node can't depend on 94 alarms at once (its table would be
astronomically large), so these treat the alarms as **conditionally independent
given the state** (Naive-Bayes): `State_t → State_t+1` and `State_t+1 → each alarm`.
Everything else is identical. They actually score a bit *worse*, most of the 94
alarms are almost always idle, so they add noise, which is why the main notebooks
use only the top 3.

## Pure DBN with `dbn.fit()` — all 94 alarms (aggregated)

**`dbn_fit_all_alarms_aggregated.ipynb`** trains a real `DynamicBayesianNetwork` with
`dbn.fit()` (and `DBNInference`) on **all 94 alarms**. Since a `State` node can't
depend on 94 alarms individually, each window's alarm activity is summarised into the
exercise's two attributes:  (**Count** and **Duration**), aggregated two ways each
(total firings / distinct alarms, and total / longest duration), discretised into four
categories. It's a demonstration that the pure `dbn.fit()` path *can* handle all 94
alarms, aggregation loses *which* alarm fired, so it scores below the per-alarm models.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

## Dataset

The dataset is **not included**. Put a semicolon-separated `dataset_exercise.csv`
in the project root with these columns:

| Column | Meaning |
|--------|---------|
| `start_alarm` | alarm start timestamp |
| `end_alarm` | alarm end timestamp |
| `alarm_id` | alarm identifier |
| `machine_state` | `Running` or `Failure` |
| `time_window` | integer index of the time window |

## Run

```bash
jupyter lab        # or: jupyter notebook
```

Open either notebook and run all cells (**Kernel → Restart & Run All**).
