import numpy as np

def calculate_confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, TN, FP, FN


def calculate_f1_score(TP, TN, FP, FN):
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall    = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1

def calculate_balanced_accuracy(TP, TN, FP, FN):
    recall      = TP / (TP + FN) if TP + FN > 0 else 0
    specificity = TN / (TN + FP) if TN + FP > 0 else 0
    return (recall + specificity) / 2

def evaluate_predictions(y_true, y_pred, verbose=True):
    TP, TN, FP, FN = calculate_confusion_matrix(y_true, y_pred)
    f1 = calculate_f1_score(TP, TN, FP, FN)
    balanced_acc = calculate_balanced_accuracy(TP, TN, FP, FN)
    
    if verbose:
        print("\n" + "=" * 40)
        print("EVALUATION RESULTS")
        print("=" * 40)
        print(f"True Positives  (TP): {TP}")
        print(f"True Negatives  (TN): {TN}")
        print(f"False Positives (FP): {FP}")
        print(f"False Negatives (FN): {FN}")
        print("-" * 40)
        print(f"F1 Score:{f1:.4f}")
        print(f"Balanced Accuracy:{balanced_acc:.4f}")
        print("=" * 40)
    return TP,TN,FP,FN,f1,balanced_acc


def stratified_k_fold_split(X, y, k=5):
    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]
    
    np.random.seed(42)

    np.random.shuffle(class_0_indices)
    np.random.shuffle(class_1_indices)

    class_0_folds = np.array_split(class_0_indices, k)
    class_1_folds = np.array_split(class_1_indices, k)
    folds = []
    for i in range(k):
        val_indices   = np.concatenate([class_0_folds[i], class_1_folds[i]])
        train_class_0 = np.concatenate([class_0_folds[j] for j in range(k) if j != i])
        train_class_1 = np.concatenate([class_1_folds[j] for j in range(k) if j != i])
        train_indices = np.concatenate([train_class_0, train_class_1])
        
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)

        folds.append((train_indices, val_indices))
    return folds


if __name__ == "__main__":
    
    print("evaluation functions to measure our model:")
    print("=" * 50)
    






