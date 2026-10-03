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
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # row-preserving cleaning + leak-safe preprocessing
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 2 | Introduction & baseline pipeline | Initial version: project structure, naive train/test split, minimal preprocessing, logistic-regression baseline, a first fairness check, train-vs-test accuracy reporting, and timestamped run reports. |
| 4 | Leak-safe preprocessing | Row-preserving cleaning; duplicate removal on labelled training data only; an MNAR indicator for configured columns; configured median/mode imputation, categorical encoding and numeric scaling inside the model Pipeline; stratified development/locked-test split; `dummy` and `random_forest` model options. |

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
- **development training accuracy and locked-test accuracy, side by side.** The test split is held out from preprocessing fits and model training.
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

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