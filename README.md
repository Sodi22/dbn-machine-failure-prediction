# DBN Machine-Failure Prediction — summary

This folder compares five Dynamic Bayesian Network (DBN) variants for predicting the next machine state from recent alarm activity.

Target:

P(State(t+1) = Failure | State(t), alarm evidence at t)

## 1) Model variants

| # | Notebook | Alarm set | Structure | Estimator | Evaluation |
|---|---|---|---|---|---|
| 1 | `dbn_failure_prediction.ipynb` | top 3 alarms | full joint CPT | BDeu smoothing | held-out last 25% |
| 2 | `dbn_alarm_data_prediction.ipynb` | top 3 alarms | real DBN | MLE | in-sample |
| 3 | `dbn_failure_prediction_all_alarms.ipynb` | all 94 alarms | Naive-Bayes style | BDeu smoothing | held-out last 25% |
| 4 | `dbn_alarm_data_prediction_all_alarms.ipynb` | all 94 alarms | Naive-Bayes style | MLE | in-sample |
| 5 | `dbn_fit_all_alarms_aggregated.ipynb` | all 94 alarms, aggregated | real DBN with summary features | MLE | in-sample |

All variants use the same pipeline:

1. compute alarm duration;
2. discretise counts/durations into categories;
3. build one row per time window;
4. form consecutive transitions `t -> t+1`;
5. train and evaluate on the next-state prediction task.

## 2) Executed score summary

Positive class: Failure.

| # | Variant | Accuracy | Precision | Recall | F1 | Confusion matrix |
|---|---|---:|---:|---:|---:|---|
| 1 | BDeu, top 3 alarms (held-out 25%) | 0.878 | 0.889 | 0.889 | 0.889 | TP=24, FP=3, FN=3, TN=19 |
| 2 | MLE, top 3 alarms (in-sample) | 0.656 | 0.696 | 0.211 | 0.325 | TP=16, FP=7, FN=60, TN=112 |
| 3 | BDeu, all 94 alarms (held-out 25%) | 0.653 | 0.857 | 0.444 | 0.585 | TP=12, FP=2, FN=15, TN=20 |
| 4 | MLE, all 94 alarms (in-sample) | 0.836 | 0.940 | 0.618 | 0.746 | TP=47, FP=3, FN=29, TN=116 |
| 5 | Pure DBN, all 94 alarms aggregated (in-sample) | 0.697 | 0.718 | 0.368 | 0.487 | TP=28, FP=11, FN=48, TN=108 |

Persistence baseline (for the held-out split): 0.816.

Interpretation:

- The best predictive model is variant 1: `dbn_failure_prediction.ipynb`.
- It is the only model evaluated on unseen data with a strong F1 and a real improvement over the persistence baseline.
- The in-sample MLE variants score more highly on accuracy, but that is inflated by fitting on the same transitions they are evaluated on.
- The all-94 variants are weaker because most alarm channels are mostly idle and therefore add noise rather than signal.

## 3) Why the models score this way

### Best model: BDeu + top 3 alarms

This variant keeps the feature set small and informative. The model sees only the three most frequent alarms and keeps a full joint state-to-next-state conditional probability table. The BDeu prior prevents zero-probability issues and keeps inference stable on the held-out partition. This combination yields the strongest F1 and the clearest improvement over persistence.

### Why the pure MLE DBN is not the winner

The MLE models are simpler and textbook-like, but they do not smooth unseen combinations. When a query hits an unseen alarm pattern, the probability mass collapses or the inference path falls back to the persistence baseline. That makes the in-sample numbers look respectable, while the actual failure recall remains weak.

### Why all-94 alarm models degrade

Most of the 94 alarms are almost always `None` or low-activity. In a Naive-Bayes-style structure, their independent contributions accumulate noise, and the model gets pulled toward the majority class (Running) rather than capturing the rare but important failure transitions. This is visible in the lower recall for the all-94 versions.

### Why aggregated all-94 DBN is weaker

The aggregated model keeps the DBN conceptually valid, but it loses the identity of the specific alarm that fired. A failure caused by one warning pattern becomes hard to separate from general noise, so the model misses many failure windows and the recall drops.

## 4) Recommended take-away

If the goal is predictive performance on unseen data, the recommendation is:

1. use the top-3 alarm subset;
2. keep the full joint structure;
3. use BDeu smoothing instead of plain MLE.

This is the most robust and interpretable DBN configuration in this project.

## 5) How to reproduce

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Then run the notebooks in this folder and compare the evaluation blocks.

More detailed per-variant notes are in `MODEL_COMPARISON.md`.
