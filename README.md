# Baseline Predictive Pipeline -- ETAI
20260674 - Marin Cepeleaga

Week 2
LR - Train accuracy: 0.679
Test accuracy:  0.678
Gap (train - test): +0.001
Generalização aparentemente muito mais estável.


DT - Train accuracy: 0.705
Test accuracy:  0.653
Gap (train - test): +0.052
Evidência de maior diferença entre treino e teste.

Por enquanto o de regressão logistica está melhor porque tem um gap menor e um test acc maior.

Week3

LR - Train accuracy: 0.676
Test accuracy: 0.673
Gap (train - test): +0.003
Train and test accuracy are still very close, suggesting stable generalization. Compared with Week 1, test accuracy is slightly lower (0.673 vs. 0.678), and the gap is slightly larger (0.003 vs. 0.001).

DT - Train accuracy: 0.736
Test accuracy: 0.647
Gap (train - test): +0.088
Training accuracy increased compared with Week 1, but test accuracy decreased (0.647 vs. 0.653) and the gap grew (0.088 vs. 0.052). This suggests the decision tree is fitting the training data more closely without improving its performance on unseen data.

Overall, the logistic regression still generalizes more consistently and has higher test accuracy than the decision tree. The Week 3 results also show lower false-positive rates than COMPAS for several larger race groups, though the very small group counts make some comparisons unreliable.

Week4

LR baseline

LR - Train accuracy: 0.676
Test accuracy:  0.658
Gap (train - test): +0.018
The logistic regression is the most stable model and performs best overall on unseen data.

DT baseline

DT - Train accuracy: 0.684
Test accuracy:  0.665
Gap (train - test): +0.020
The decision tree is slightly better than the dummy model, but it is less stable than logistic regression.

RF baseline

RF - Train accuracy: 0.690
Test accuracy:  0.669
Gap (train - test): +0.021
The random forest is close to logistic regression, but it still shows a slightly larger gap and does not improve generalization meaningfully.

Dummy baseline

Dummy - Train accuracy: 0.549
Test accuracy:  0.550
Gap (train - test): -0.000
The dummy model performs at the majority-class baseline and has no real learning signal.

Week 5/6

LR - Train accuracy: 0.676
Test accuracy:  0.661
Gap (train - test): +0.015
The logistic regression remains the most stable model, with a low train-test gap and strong generalization on unseen data.

DT - Train accuracy: 0.687
Test accuracy:  0.670
Gap (train - test): +0.018
The decision tree improves over the baseline and reaches a stronger test accuracy, but it is still less stable than logistic regression.

RF - Train accuracy: 0.689
Test accuracy:  0.673
Gap (train - test): +0.016
The random forest gives the best standard ensemble performance, with a slightly higher test accuracy than logistic regression while staying reasonably stable.

Oblique RF - Train accuracy: 0.695
Test accuracy:  0.674
Gap (train - test): +0.021
The oblique random forest reaches the strongest test accuracy overall, though its gap is a bit larger than the other models.

AdaBoost - Train accuracy: 0.682
Test accuracy:  0.672
Gap (train - test): +0.010
The AdaBoost model is very stable and competitive, with one of the lowest overfitting gaps while keeping test performance close to the best models.

Across the tuned models, the false-positive rates for the larger race groups are consistently lower than COMPAS, while the ensemble approaches offer the highest predictive accuracy without a large drop in fairness.

Week 7

Dummy baseline

Dummy - Train accuracy: 0.508
Test accuracy:  0.521
Gap (train - test): -0.013
The dummy model remains close to chance, with a nested-CV ROC-AUC of 0.500, and provides a baseline for judging whether the other models learn useful patterns.

LR

LR - Train accuracy: 0.679
Test accuracy:  0.661
Gap (train - test): +0.017
Logistic regression performs better than the dummy baseline, but has the lowest test accuracy among the trained models.

DT

DT - Train accuracy: 0.673
Test accuracy:  0.676
Gap (train - test): -0.003
The decision tree has a small negative train-test gap, indicating similar performance on the training and test sets, though its test accuracy is lower than the ensemble models.

RF

RF - Train accuracy: 0.685
Test accuracy:  0.680
Gap (train - test): +0.005
The random forest gives the second-highest test accuracy, with only a small train-test gap.

Oblique RF

Oblique RF - Train accuracy: 0.687
Test accuracy:  0.670
Gap (train - test): +0.017
The oblique random forest has a larger train-test gap, but ties for the highest nested-CV ROC-AUC (0.727).

AdaBoost

AdaBoost - Train accuracy: 0.684
Test accuracy:  0.673
Gap (train - test): +0.012
AdaBoost is competitive with the other ensemble models and also ties for the highest nested-CV ROC-AUC (0.727).

Histogram gradient boosting

Hist. gradient boosting - Train accuracy: 0.685
Test accuracy:  0.681
Gap (train - test): +0.004
Histogram gradient boosting reaches the highest test accuracy and maintains a small train-test gap, suggesting stable generalization.

Overall, histogram gradient boosting performs best on the locked test set, while oblique random forest and AdaBoost have the highest nested-CV ROC-AUC. The best model therefore depends on the evaluation metric. At the 0.50 threshold, histogram gradient boosting has lower test-set false-positive rates than COMPAS for the African-American, Caucasian, and Hispanic groups, and the same rate for the Other group. Results for the Asian and Native American groups are based on only 2 and 1 test examples, respectively, and should not be treated as reliable comparisons.

Week8

The tuned XGBoost model achieved nested-CV ROC-AUC of 0.726 ± 0.011 with feature selection and 0.727 ± 0.013 without it; both runs had locked-test accuracy of 0.674. Using the five displayed fold scores and a test/train ratio of 1/4, the corrected paired test gives a mean ROC-AUC difference (selection minus no selection) of -0.001 (`t = -0.58`, `p ≈ 0.59`), so the small score difference is not statistically significant. Feature selection kept just age, priors_count, and priors_per_adult_year, compared with all 13 preprocessed features in the run without selection. Because the compact model performs essentially as well, I would keep selection enabled for simplicity, while noting that these five folds do not show a measurable performance gain.

Week 10

On the same five COMPAS development folds with tuning disabled, XGBoost achieved mean ROC-AUC of 0.7190 with median, 0.7188 with KNN, 0.7202 with iterative imputation, and 0.7201 with random-forest imputation. The corrected paired tests gave p-values of 0.415 to 0.907 against median, so I kept median for its faster fit. With median, training accuracy was 0.705 and locked-test accuracy was 0.678, a gap of +0.027. The saved pipeline scored all 20 rows in the new COMPAS file.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── predict.py              # score new CSV rows with the saved pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # row-preserving cleaning + leak-safe preprocessing
│   ├── model.py             # model construction and complete preprocessing/model Pipeline
│   ├── tuning.py            # cross-validation, Optuna tuning, and nested CV
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
├── models/                 # fitted pipelines saved by training
└── data/
    ├── compas_two_year_recidivism.csv
    ├── compas_new_people.csv
    ├── adult_train.csv       # option A training data
    ├── adult_new.csv         # option A unseen rows for inference
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 2 | Introduction & baseline pipeline | Initial version: project structure, naive train/test split, minimal preprocessing, logistic-regression baseline, a first fairness check, train-vs-test accuracy reporting, and timestamped run reports. |
| 4 | Leak-safe preprocessing | Row-preserving cleaning; duplicate removal on labelled training data only; an MNAR indicator for configured columns; configured median/mode imputation, categorical encoding and numeric scaling inside the model Pipeline; stratified development/locked-test split; `dummy` and `random_forest` model options. |
| 5 | Hyperparameter tuning | Stratified CV of the complete pipeline; Optuna search on development data; nested CV for an honest tuned-model estimate; final refit after tuning; out-of-fold classification and fairness reports. |
| 6 | Ensembles, feature engineering & boosting | AdaBoost, oblique random forest, gradient boosting, histogram gradient boosting, and a Naive Bayes baseline; fixed-formula features with a configuration switch; per-model search spaces support nested tuning. |
| 7 | Metrics and model comparison | Out-of-fold probabilities; accuracy, balanced accuracy, precision, recall, F1, F-beta, ROC-AUC, and average precision; a configurable selection metric for tuning; probability-thresholded fairness reports and confusion matrices; and a persistent CSV scoreboard with the dummy baseline and each run. |
| 8 | XGBoost & feature selection | XGBoost support and tuning; leak-safe configurable feature selectors (`vote`, `kbest`, `from_model`) inside the model pipeline; nested tuning of the vote threshold; per-run pipeline caching; and selected-feature reporting. |
| 9 | Beyond COMPAS (option A) | Added an Adult-income config (`config_adult.yaml`), config-file selection in `main.py`, dataset-agnostic handling for engineered features and fairness reporting, and export of the `adult_new.csv` predictions. |
| 10 | Missing values and deployment | Configurable numeric imputation after scaling, imputer tracking in the scoreboard and run reports, and a saved fitted pipeline with a row-preserving prediction command. |

## Beyond COMPAS — Option A: Adult income

Run it from the repo root:
```bash
python main.py config_adult.yaml
```

What broke, why, and how it was fixed:

| Breakage | Why it broke | Fix |
|----------|--------------|-----|
| Hard-coded COMPAS columns | The pipeline assumed `priors_count`, `juv_*`, and `score_text` existed everywhere | Generalised engineered-feature creation to skip missing COMPAS-only columns, and made fairness reporting handle datasets without `score_text` |
| Sensitive attribute forced into the model | `sex` was still included in the feature list for Adult, even though it should be audited only | Excluded the configured sensitive attr from `preprocessing.categorical_features` while keeping it for fairness reporting |
| One config file for every dataset | The project always loaded `config.yaml` | `main.py` accepts an optional config path and defaults to `config.yaml` |
| New-data prediction was not implemented | Adult required scoring the 2,000 unseen rows without dropping or labelling them | Added `data.new_path` handling and CSV export of `id`, probability, and prediction |

Adult run (development-set out-of-fold estimates):

| Metric | Target | Achieved |
|--------|--------|----------|
| ROC-AUC | >= 0.90 | 0.905 |
| Accuracy | >= 0.855 | 0.856 |
| F1 (class 1, threshold 0.5) | >= 0.66 | 0.67 |

The model kept the Adult features that matter (numeric census variables, workclass, marital status, occupation, relationship, race, native country, and the workclass/occupation missingness indicators) and produced `results/adult/adult_new_predictions.csv` with all 2,000 rows scored. The hardest assumption to catch was that the original pipeline had a few COMPAS-specific columns and metrics baked in, especially the fairness comparison that expected `score_text` to exist. Once those assumptions were made dataset-agnostic, the same code path worked cleanly for both COMPAS and Adult.

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, cleans the data without dropping rows, removes duplicate records from the labelled training data, creates a stratified development/locked-test split, and fits preprocessing and the model together. It prints:
- per-fold and mean/std results for every metric in `cv.metrics`, with
  `cv.selection_metric` used for tuning and ranking
- the same-fold dummy baseline and, when `tuning.enabled` is true,
  nested-CV results and the Optuna tuning report
- out-of-fold classification, confusion-matrix, and fairness reports on
  the development split
- **development training accuracy and locked-test accuracy, side by side.** The test split is held out from preprocessing fits and model training.
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score
- the features retained by the configured selector

The full report is also saved to a timestamped file in `results/`.
Every run is appended to `results/runs.csv`; the terminal and report
show comparable runs ranked by the configured selection metric. To
choose a model, set `model.type` and its `params` in `config.yaml`.
Feature selection is configured under `feature_selection`; it is enabled by
default with the combined vote selector and can be switched to `kbest` or
`from_model`. XGBoost is the default model and requires the `xgboost` dependency
listed in `requirements.txt`.
Set `feature_engineering.enabled` to `false` to run the same CV
comparisons without engineered features. Use a smaller
`tuning.n_trials` while checking setup; restore the intended trial
count for final experiments.
`results/` is created automatically and isn't tracked in git (see
`.gitignore`) since it contains generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Missing values

The numeric pipeline scales observed values before filling gaps. To reproduce the comparison, set `tuning.enabled` to `false` and run `python main.py` once per `preprocessing.imputation.numeric_strategy`. I used the same five stratified development folds and XGBoost settings for all four runs, then applied `corrected_paired_test()` to the fold ROC-AUC scores with `test_train_ratio=1/4`.

| Imputer | Mean CV ROC-AUC | Corrected p-value vs. median |
|---------|----------------:|-----------------------------:|
| `median` | 0.7190 | — |
| `knn` | 0.7188 | 0.907 |
| `iterative` | 0.7202 | 0.658 |
| `random_forest` | 0.7201 | 0.415 |

Iterative imputation had the highest mean ROC-AUC, ahead of median by 0.0012. The corrected p-values versus median ranged from 0.415 to 0.907, so I kept median for its faster fit. Both iterative strategies reached the configured 10-iteration limit in some fits.

## Predicting new data

Run training and prediction from the project root in the same Python environment.

```bash
python main.py
python predict.py data/compas_new_people.csv predictions.csv
```

Training saves the final fitted pipeline to `output.model_path` in the config (`models/model.joblib` for COMPAS). The sample command writes 20 rows with `id`, the positive-class `probability`, and the binary `prediction` at a threshold of 0.5. It preserves the input order and duplicate IDs. The input CSV can omit the target column.

For Adult income, `python main.py config_adult.yaml` saves `models/adult_model.joblib`. Use `python predict.py data/adult_new.csv predictions.csv --config config_adult.yaml` to score the new Adult rows with that model.

## Model evaluation

To compare two runs on the same development-set folds, pass their
`fold_1` ... `fold_5` values for the same metric to
`src.evaluate.corrected_paired_test`. For the default five-fold CV, use
`test_train_ratio=1/4`. This applies the Nadeau-Bengio correction for the
overlap between training folds; use matched data, folds, and metrics.

```python
from src.evaluate import corrected_paired_test

result = corrected_paired_test(
    selected_run[["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]],
    baseline_run[["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]],
    test_train_ratio=1 / 4,
)
print(result)
```

## Dataset

See `data/README.md`.


    STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

Increase the number of iterations to improve the convergence (max_iter=1000).
You might also want to scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
Train accuracy: 0.679
Test accuracy:  0.678
Gap (train - test): +0.001

Classification report (test set):
              precision    recall  f1-score   support

           0       0.69      0.74      0.72       684
           1       0.66      0.60      0.63       568

    accuracy                           0.68      1252
   macro avg       0.68      0.67      0.67      1252
weighted avg       0.68      0.68      0.68      1252

False positive rate by race
(share of people who did NOT reoffend, but were predicted to)

  Our model:
     African-American    FPR = 0.50  (n=6)
     Caucasian           FPR = 0.00  (n=1)
    -                    FPR = 0.20  (n=5)
    ?                    FPR = 0.33  (n=3)
    AFRICAN-AMERICAN     FPR = 0.00  (n=4)
    African American     FPR = 0.33  (n=3)
    African-American     FPR = 0.33  (n=303)
    Asian                FPR = 0.33  (n=3)
    CAUCASIAN            FPR = 0.25  (n=4)
    Caucasian            FPR = 0.24  (n=232)
    Hispanic             FPR = 0.10  (n=61)
    Native American      FPR = 0.00  (n=2)
    Other                FPR = 0.15  (n=41)
    african-american     FPR = 0.10  (n=10)
    caucasian            FPR = 0.00  (n=3)
    hispanic             FPR = 0.33  (n=3)

  COMPAS's own score:
     African-American    FPR = 0.50  (n=6)
     Caucasian           FPR = 0.00  (n=1)
    -                    FPR = 0.20  (n=5)
    ?                    FPR = 0.00  (n=3)
    AFRICAN-AMERICAN     FPR = 0.25  (n=4)
    African American     FPR = 0.33  (n=3)
    African-American     FPR = 0.44  (n=303)
    Asian                FPR = 0.00  (n=3)
    CAUCASIAN            FPR = 0.25  (n=4)
    Caucasian            FPR = 0.25  (n=232)
    Hispanic             FPR = 0.15  (n=61)
    Native American      FPR = 0.50  (n=2)
    Other                FPR = 0.20  (n=41)
    african-american     FPR = 0.50  (n=10)
    caucasian            FPR = 0.00  (n=3)
    hispanic             FPR = 0.33  (n=3)

Full results saved to results\run_20260916_091409.txt

Full results saved to results\run_20260916_092131.txt
(ETAI) PS C:\Users\marin\ETAI-Pipeline> python main.py
Train accuracy: 0.705
Test accuracy:  0.653
Gap (train - test): +0.052

Classification report (test set):
              precision    recall  f1-score   support

           0       0.69      0.68      0.68       684
           1       0.62      0.63      0.62       568

    accuracy                           0.65      1252
   macro avg       0.65      0.65      0.65      1252
weighted avg       0.65      0.65      0.65      1252

False positive rate by race
(share of people who did NOT reoffend, but were predicted to)

  Our model:
     African-American    FPR = 0.50  (n=6)
     Caucasian           FPR = 1.00  (n=1)
    -                    FPR = 0.40  (n=5)
    ?                    FPR = 0.67  (n=3)
    AFRICAN-AMERICAN     FPR = 0.00  (n=4)
    African American     FPR = 0.33  (n=3)
    African-American     FPR = 0.39  (n=303)
    Asian                FPR = 0.67  (n=3)
    CAUCASIAN            FPR = 0.25  (n=4)
    Caucasian            FPR = 0.29  (n=232)
    Hispanic             FPR = 0.18  (n=61)
    Native American      FPR = 0.00  (n=2)
    Other                FPR = 0.22  (n=41)
    african-american     FPR = 0.20  (n=10)
    caucasian            FPR = 0.33  (n=3)
    hispanic             FPR = 0.33  (n=3)

  COMPAS's own score:
     African-American    FPR = 0.50  (n=6)
     Caucasian           FPR = 0.00  (n=1)
    -                    FPR = 0.20  (n=5)
    ?                    FPR = 0.00  (n=3)
    AFRICAN-AMERICAN     FPR = 0.25  (n=4)
    African American     FPR = 0.33  (n=3)
    African-American     FPR = 0.44  (n=303)
    Asian                FPR = 0.00  (n=3)
    CAUCASIAN            FPR = 0.25  (n=4)
    Caucasian            FPR = 0.25  (n=232)
    Hispanic             FPR = 0.15  (n=61)
    Native American      FPR = 0.50  (n=2)
    Other                FPR = 0.20  (n=41)
    african-american     FPR = 0.50  (n=10)
    caucasian            FPR = 0.00  (n=3)
    hispanic             FPR = 0.33  (n=3)

Full results saved to results\run_20260916_092155.txt
