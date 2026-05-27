import numpy as np

from data_handler import process_all_training_data

def get_first_fall_index(y):
    
    for i in range(len(y)):
        if y[i] == 1:
            return i
    return len(y)


def undersample_majority(X, y):
    
    first_fall=get_first_fall_index(y)

    print(f"First fall at window index: {first_fall}")

    X_before=X[:first_fall]
    y_before=y[:first_fall]

    X_after=X[first_fall:]
    y_after=y[first_fall:]

    # Find indices of each class
    class_0_indices = np.where(y_after == 0)[0]
    class_1_indices = np.where(y_after == 1)[0]
    
    num_class_0 = len(class_0_indices)
    num_class_1 = len(class_1_indices)
    
    print(f"Before balancing:")
    print(f"Class 0: {num_class_0}")
    print(f"Class 1: {num_class_1}")
    

    if num_class_0 > num_class_1:
        majority_indices=class_0_indices
        minority_indices=class_1_indices
    else:
        majority_indices=class_1_indices
        minority_indices=class_0_indices

    num_minority = len(minority_indices)
    
    np.random.seed(42)

    selected_majority = np.random.choice(
        majority_indices,
        size=num_minority,
        replace=False   ##no duplicates
    )
    
    # Combine selected class 0 with ALL class 1
    # selected_indices = np.concatenate([selected_class_0_indices, class_1_indices])
    selected_indices = np.concatenate([selected_majority, minority_indices])
    # Shuffle the combined indices
    np.random.shuffle(selected_indices)
    
    # Extract balanced data
    X_balanced_after = X_after[selected_indices]
    y_balanced_after= y_after[selected_indices]
    
    #untouched before-region + balanced after-region
    X_final=np.vstack([X_before,X_balanced_after])
    y_final = np.concatenate([y_before, y_balanced_after])

    print(f"\nAfter balancing:")
    print(f"  Class 0: {sum(y_final == 0)}")
    print(f"  Class 1: {sum(y_final == 1)}")
    print(f"  Total samples: {len(y_final)}")
    
    return X_final, y_final



def simple_smote(X, y, k=5):
    
    X = np.array(X)
    y = np.array(y)

    class_0_idx = np.where(y == 0)[0]
    class_1_idx = np.where(y == 1)[0]

    # Detect minority class
    if len(class_0_idx) > len(class_1_idx):
        minority_idx = class_1_idx
        majority_count = len(class_0_idx)
    else:
        minority_idx = class_0_idx
        majority_count = len(class_1_idx)

    X_minority = X[minority_idx]

    synthetic_samples = []

    np.random.seed(42)


    ##Keep generating synthetic samples until minority count = majority count.
    while len(X_minority) + len(synthetic_samples) < majority_count:

        i = np.random.randint(len(X_minority))
        j = np.random.choice([x for x in range(len(X_minority)) if x != i])

        x1 = X_minority[i]
        x2 = X_minority[j]

        alpha = np.random.rand()

        synthetic = x1 + alpha * (x2 - x1)

        synthetic_samples.append(synthetic)

    
    if len(synthetic_samples) == 0:
        print("Classes already balanced, no synthetic samples needed.")
        return X, y 
    
    X_syn = np.array(synthetic_samples)

    y_syn = np.ones(len(X_syn)) if len(class_1_idx) < len(class_0_idx) else np.zeros(len(X_syn))

    X_balanced = np.vstack([X, X_syn])
    y_balanced = np.concatenate([y, y_syn])

    print("After SMOTE balancing:")
    print(f"Class 0: {np.sum(y_balanced==0)}")
    print(f"Class 1: {np.sum(y_balanced==1)}")

    return X_balanced, y_balanced


if __name__ == "__main__":
    print("DATA IMBALANCE HANDLER")


    