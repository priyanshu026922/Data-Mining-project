# Fall Detection from Wearable Sensor Data using Machine Learning

---

## About the project-

FallGuard detects whether a person has fallen using time series sensor data.
It reads wearable sensor recordings, processes them into windows, and classifies
each window as a fall (Class 1) or normal activity (Class 0).

---

## Project Structure

```
FallGuard/
│
├── main.py               # Main pipeline — run this
├── data_handler.py       # Reads CSV files, creates sliding windows
├── imbalance_handler.py  # Handles class imbalance (undersampling)
├── models.py             # KNN, Logistic Regression, Decision Tree, Random Forest, XGBoost
├── evaluation.py         # F1 Score, Balanced Accuracy, Confusion Matrix, K-Fold CV
├── visualization.py      # Saves plots to output_plots/
├── model_saver.py        # Saves and loads trained models
│
├── Data/
│   ├── Sample_Training/  # Training data (subfolders of CSV files)
│   └── Sample_Test/      # Testing data (subfolders of CSV files)
│
├── saved_models/         # Trained models saved here (auto created)
└── output_plots/         # All plots saved here (auto created)
```

---

## How to Run

**1. Install dependencies:**
```
pip install numpy pandas matplotlib xgboost
```

**2. Place your data:**
```
Data/Sample_Training/  ← training CSV files
Data/Sample_Test/      ← testing CSV files
```

**3. Run the pipeline:**
```
python main.py
```

---

## What happens when you run it?

```
STEP 1 → Load and normalize sensor data
STEP 2 → Balance training data (undersample majority class)
STEP 3 → Tune hyperparameters using Stratified 5-Fold Cross Validation
STEP 4 → Train all models (without balancing and with balancing)
STEP 5 → Evaluate all models on test data
STEP 6 → Print results table and save plots
```

> On second run, saved models and best parameters are loaded automatically — no retraining needed.

---

## Models Used

| Model | Type |
|---|---|
| K-Nearest Neighbors | Distance based |
| Logistic Regression | Linear classifier |
| Decision Tree | Tree based |
| Random Forest | Ensemble (from scratch) |

KNN, Logistic Regression, Decision Tree and Random Forest are implemented from scratch using NumPy only.(Reference scratch codes-GITHUB Public Repos)

---

## Key Concepts

**Sliding Windows** — sensor data is split into overlapping windows of 50 rows with step size 25.
Each window is labeled as fall or no fall using the >40% rule.

**Class Imbalance** — falls are rare events. We undersample the majority class (normal activity)
while keeping all minority class (fall) samples intact. Sampling is applied only after
the first fall is encountered in the data.

**Normalization** — Z-score normalization applied to all features using training data statistics only.

**Hyperparameter Tuning** — Stratified 5-Fold Cross Validation used to find best parameters for each model.

---

## Output

After running, check these folders:

`output_plots/` contains:
- `class_distribution.png` — before vs after balancing
- `f1_comparison.png` — F1 scores across all models
- `cm_*.png` — confusion matrix for each model
- `cv_scores.png` — cross validation fold scores

`saved_models/` contains:
- Trained model files (`.pkl`)
- Best hyperparameters (`best_params.pkl`)

---

## Evaluation Metrics

- **F1 Score** — balances precision and recall, good for imbalanced data
- **Balanced Accuracy** — average of sensitivity and specificity

---

## Team
Data Mining Project — Fall Detection System