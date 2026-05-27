import numpy as np
import os, pickle
from data_handler import process_all_training_data, process_all_testing_data

from visualization import generate_all_plots
from model_saver import load_all_models, models_exist, save_model, SAVE_DIR
from imbalance_handler import undersample_majority, simple_smote

from evaluation import evaluate_predictions, stratified_k_fold_split

from models import (
    KNearestNeighbors,
    LogisticRegression,
    SimpleDecisionTree,
    SimpleRandomForest
)


# ──-------- Hyperparameter grids ──────────────────────────────────────────────────────
KNN_PARAMS = [{'k': 3}, {'k': 5}, {'k': 7}, {'k': 11}]
LR_PARAMS  = [{'learning_rate': 0.1,  'num_iterations': 500},
              {'learning_rate': 0.01, 'num_iterations': 500},
              {'learning_rate': 0.01, 'num_iterations': 1000},
              {'learning_rate': 0.001,'num_iterations': 1000}]
DT_PARAMS  = [{'max_depth': 3}, {'max_depth': 5},
              {'max_depth': 7}, {'max_depth': 10}]
RF_PARAMS  = [{'n_trees': 5,  'max_depth': 3}, {'n_trees': 10, 'max_depth': 5},
              {'n_trees': 10, 'max_depth': 7}, {'n_trees': 15, 'max_depth': 5}]

# Filenames used when saving/loading
MODEL_NAMES_NO_BAL = ['knn_nobal', 'lr_nobal', 'dt_nobal', 'rf_nobal']
MODEL_NAMES_BAL    = ['knn_bal',   'lr_bal',   'dt_bal',   'rf_bal']


def make_model(model_name, params):
    if model_name == 'KNN': return KNearestNeighbors(**params)
    if model_name == 'LR':  return LogisticRegression(**params)
    if model_name == 'DT':  return SimpleDecisionTree(**params)
    if model_name == 'RF':  return SimpleRandomForest(**params)
    
    raise ValueError(f"Unknown model name: {model_name}")

def cross_val_score_f1(model_name, params, X, y, n_folds=5):
    folds = stratified_k_fold_split(X, y, k=n_folds)
    f1_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        model = make_model(model_name, params)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])
        _, _, _, _, f1, _ = evaluate_predictions(y[val_idx], y_pred, verbose=False)
        f1_scores.append(f1)
        print(f"    Fold {fold_idx+1}/{n_folds} → F1: {f1:.4f}")
    mean_f1 = float(np.mean(f1_scores))
    print(f"Mean F1: {mean_f1:.4f}  params={params}")
    return mean_f1, f1_scores


def tune_model(model_name, param_grid, X, y, n_folds=5):

    print(f"\n  Tuning {model_name} – {len(param_grid)} combos:{n_folds} folds\n")
    best_f1, best_params, best_fold_scores = -1, None, []
    for params in param_grid:
        print(f"  Testing: {params}")
        mean_f1, fold_scores = cross_val_score_f1(model_name, params, X, y, n_folds)
        if mean_f1 > best_f1:
            best_f1, best_params, best_fold_scores = mean_f1, params, fold_scores
    print(f"\n Best {model_name}: {best_params}(F1={best_f1:.4f})")
    return best_params, best_f1, best_fold_scores


def train_and_save(X_train, y_train, X_balanced, y_balanced,
                   best_knn, best_lr, best_dt, best_rf):
    """Trains all 8 models (4 unbalanced + 4 balanced) and saves them."""

    #Train and save models WITHOUT balancing:
    print("\n  Training & saving models WITHOUT balancing...")

    models_nobal = {
        'knn_nobal': KNearestNeighbors(**best_knn),
        'lr_nobal':  LogisticRegression(**best_lr),
        'dt_nobal':  SimpleDecisionTree(**best_dt),
        'rf_nobal':  SimpleRandomForest(**best_rf),
    }
    for name, model in models_nobal.items():
        print(f"\n---{name}")
        model.fit(X_train, y_train)
        save_model(model,name)

    # Train and save models WITH balancing:-
    print("\nTraining & saving models WITH balancing...")
    models_bal = {
        'knn_bal': KNearestNeighbors(**best_knn),
        'lr_bal':  LogisticRegression(**best_lr),
        'dt_bal':  SimpleDecisionTree(**best_dt),
        'rf_bal':  SimpleRandomForest(**best_rf),
    }

    for name, model in models_bal.items():
        print(f"\n  → {name}")
        model.fit(X_balanced, y_balanced)
        save_model(model, name)

    return models_nobal, models_bal


def evaluate_models(models_nobal, models_bal,
                    X_test, y_test,
                    best_knn, best_lr, best_dt, best_rf):
    """Runs evaluate_predictions on all trained/loaded models."""

    label_map = {
        'knn_nobal':f"KNN {best_knn}", 'lr_nobal': f"LR {best_lr}",
        'dt_nobal':f"DT {best_dt}",'rf_nobal': f"RF {best_rf}",
        'knn_bal':f"KNN {best_knn}", 'lr_bal':   f"LR {best_lr}",
        'dt_bal':f"DT {best_dt}",  'rf_bal':   f"RF {best_rf}",
    }

    def _eval_group(models_dict):
        results = []
        for fname, model in models_dict.items():
            print(f"\n{'='*60}\nMODEL: {label_map[fname]}\n{'='*60}")
            y_pred = model.predict(X_test)
            TP, TN, FP, FN, f1, bal_acc = evaluate_predictions(y_test, y_pred, verbose=True)
            results.append({'model': label_map[fname],
                            'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
                            'F1': f1, 'Balanced_Accuracy': bal_acc})
        return results

    results_without = _eval_group(models_nobal)
    results_with    = _eval_group(models_bal)
    return results_without, results_with


def print_results_table(results, title):
    print(f"\n{'#'*70}\n  {title}\n{'#'*70}")
    print(f"\n{'Model':<25} {'TP':<6} {'TN':<6} {'FP':<6} {'FN':<6} {'F1':<8} {'Bal Acc':<8}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<25} {r['TP']:<6} {r['TN']:<6} {r['FP']:<6} {r['FN']:<6} "
              f"{r['F1']:<8.4f} {r['Balanced_Accuracy']:<8.4f}")


def main():
    print("  TIME SERIES FALL DETECTION – FULL PIPELINE")
    print("="*70)

    # Load data ::
    print("\n" + "="*70)
    print("STEP 1: Loading Data")
    print("="*70)

    training_folder = r"Data\Sample_Training"
    testing_folder  = r"Data\Sample_Test"


    print("\nProcessing training data...")
    X_train, y_train = process_all_training_data(training_folder)
    print("\nProcessing testing data...")
    X_test, y_test = process_all_testing_data(testing_folder)
    
    mean = np.mean(X_train, axis=0)
    std  = np.std(X_train, axis=0)
    
    std[std == 0] = 1 

    X_train = (X_train - mean) / std 
    X_test  = (X_test  - mean) / std 


    print(f"\nTraining: {X_train.shape} | Testing: {X_test.shape}")
    print(f"Train ---> Class 0: {sum(y_train==0)}  Class 1: {sum(y_train==1)}")
    print(f"Test  ---> Class 0: {sum(y_test==0)}   Class 1: {sum(y_test==1)}")

    # Balance:-
    print("\n" + "="*70)
    print("STEP 2: Balancing Training Data")
    print("="*70)

    X_balanced, y_balanced  = undersample_majority(X_train, y_train)
    print(f"After undersampling → Class 0: {sum( y_balanced ==0)}  Class 1: {sum( y_balanced ==1)}")

    
    # ── 3. Hyperparameter tuning OR loading best params ──────────────────────────
    PARAMS_FILE = os.path.join(SAVE_DIR, "best_params.pkl")

    if os.path.exists(PARAMS_FILE):
        # load saved best params:
        print("\n" + "="*70)
        print("STEP 3: Loading saved best hyperparameters")
        print("="*70)

        with open(PARAMS_FILE, 'rb') as f:
            saved = pickle.load(f)
        best_knn   = saved['knn']
        best_lr    = saved['lr']
        best_dt    = saved['dt']
        best_rf    = saved['rf']
        cv_history = saved.get('cv_history', {})
        print(f"  KNN: {best_knn}")
        print(f"  LR:  {best_lr}")
        print(f"  DT:  {best_dt}")
        print(f"  RF:  {best_rf}")

    else:
        # tune and save best params
        print("\n" + "="*70)
        print("STEP 3: Hyperparameter Tuning (Stratified 5-Fold CV)")
        print("="*70)
        N_FOLDS = 5
        best_knn, _, knn_cv = tune_model('KNN', KNN_PARAMS, X_balanced, y_balanced, N_FOLDS)
        best_lr,  _, lr_cv  = tune_model('LR',  LR_PARAMS,  X_balanced, y_balanced, N_FOLDS)
        best_dt,  _, dt_cv  = tune_model('DT',  DT_PARAMS,  X_balanced, y_balanced, N_FOLDS)
        best_rf,  _, rf_cv  = tune_model('RF',  RF_PARAMS,  X_balanced, y_balanced, N_FOLDS)

        cv_history = {
            f"KNN {best_knn}": knn_cv,
            f"LR {best_lr}":   lr_cv,
            f"DT {best_dt}":   dt_cv,
            f"RF {best_rf}":   rf_cv,
        }

        os.makedirs("saved_models", exist_ok=True)
        with open(PARAMS_FILE,'wb') as f:
            pickle.dump({'knn': best_knn, 'lr': best_lr,
                         'dt': best_dt,  'rf': best_rf,
                         'cv_history': cv_history}, f)
        print(f"\n Best params saved to {PARAMS_FILE}")


    # Train models OR load from disk :
    all_saved = models_exist(MODEL_NAMES_NO_BAL + MODEL_NAMES_BAL)

    if all_saved:
        print("Loading saved models (Skipping retraining)")
        print("="*70)
        loaded= load_all_models(MODEL_NAMES_NO_BAL + MODEL_NAMES_BAL)
        models_nobal = {k: loaded[k] for k in MODEL_NAMES_NO_BAL}
        models_bal   = {k: loaded[k] for k in MODEL_NAMES_BAL}

    else:
        # FIRST RUN: train and save
        print("\n" + "="*70)
        print("Training and Saving Models")

        models_nobal, models_bal = train_and_save(
            X_train, y_train, X_balanced, y_balanced,
            best_knn, best_lr, best_dt, best_rf)

    # Evaluate:
    print("\n" + "="*70)
    print("STEP 6: Evaluating on Test Set")
    print("="*70)
    results_without, results_with = evaluate_models(   
        models_nobal, models_bal, X_test, y_test,
        best_knn, best_lr, best_dt, best_rf)

    # ──────────────────────────────────────────────────────────
    print_results_table(results_without,"RESULTS WITHOUT BALANCING")
    print_results_table(results_with,"RESULTS WITH BALANCING")

    # ─────────────────────────────────────────────────────────────────
    generate_all_plots(results_without, results_with, y_train, y_balanced, cv_history)

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print(f"Models saved in: {SAVE_DIR}/")
    print("Plots saved in:  output_plots/")
    print("="*70)


if __name__ == "__main__":
    main()



