import os
import numpy as np
import matplotlib
matplotlib.use('Agg') #to directly save the images to file
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


#CLASS DISTRIBUTION-----
# Shows before vs after balancing
def plot_class_distribution(y_train, y_balanced):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, y, title in zip(
        axes,
        [y_train, y_balanced],
        ["Before Balancing", "After Balancing"]
    ):
        counts = [np.sum(y == 0), np.sum(y == 1)]
        ax.bar(["Class 0", "Class 1"], counts, color=["steelblue", "tomato"])
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.set_xlabel("Class")

    fig.suptitle("Class Distribution", fontsize=14)
    path = os.path.join(OUTPUT_DIR, "class_distribution.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print("  Class distribution saved")


#CONFUSION MATRIX
def plot_confusion_matrix(TP, TN, FP, FN, name):
    cm = np.array([[TN, FP], [FN, TP]])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)

    # Numbers inside each cell
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=14)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted 0", "Predicted 1"])
    ax.set_yticklabels(["Actual 0", "Actual 1"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title(f"Confusion Matrix - {name}")

    clean_name = name.replace("'", "").replace("{", "").replace("}", "") \
                     .replace(":", "").replace(" ", "_").replace(",", "")

    path = os.path.join(OUTPUT_DIR, f"cm_{clean_name}.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()


#F1 COMPARISON---
#Compare model without and with balancing
def plot_f1_comparison(results_without, results_with):
    names = [r['model'] for r in results_without]
    f1_wo = [r['F1'] for r in results_without]
    f1_w  = [r['F1'] for r in results_with]

    x = np.arange(len(names))

    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, f1_wo, 0.4, label="Without Balancing", color="steelblue")
    plt.bar(x + 0.2, f1_w,  0.4, label="With Balancing",    color="tomato")

    plt.xticks(x, names, rotation=15)
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.title("F1 Score: Without vs With Balancing")
    plt.legend()

    path = os.path.join(OUTPUT_DIR, "f1_comparison.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print("  F1 comparison saved")


#CV FOLD SCORES----
#Shows performance consistency across folds
def plot_cv_scores(cv_history):

    plt.figure(figsize=(8, 5))
    for name, scores in cv_history.items():
        plt.plot(range(1, len(scores)+1), scores, marker='o', label=name)

    plt.title("Cross Validation F1 Scores per Fold")
    plt.xlabel("Fold")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.legend()

    path = os.path.join(OUTPUT_DIR, "cv_scores.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print("  CV scores saved")


# MAIN FUNCTION
def generate_all_plots(results_without, results_with, y_train, y_balanced, cv_history=None):
    print("\nGenerating plots...")

    plot_class_distribution(y_train, y_balanced)

    plot_f1_comparison(results_without, results_with)

    for r in results_with:
        plot_confusion_matrix(r['TP'], r['TN'], r['FP'], r['FN'], r['model'])
        print(f"  Confusion matrix saved for {r['model']}")

    if cv_history:
        plot_cv_scores(cv_history)

    print("\nAll plots saved in output_plots/")