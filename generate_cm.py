import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import argparse
from pathlib import Path
import glob

def plot_confusion_matrix(results_path: Path):
    """
    Calculates the confusion matrix for each fold, then takes the mean
    and std deviation between folds and plots the result.
    """
    pred_files = sorted(glob.glob(str(results_path / "data" / "predictions_fold_*.npy")))
    target_files = sorted(glob.glob(str(results_path / "data" / "targets_fold_*.npy")))

    if not pred_files or not target_files or len(pred_files) != len(target_files):
        print(f"Erro: Predict/target files not found or inconsistent in {results_path / 'data'}")
        return

    normalized_cms = []
    class_labels = [0, 1, 2, 3]

    # Calculates the normalized confusion matrix for each fold
    for pred_file, target_file in zip(pred_files, target_files):
        preds = np.load(pred_file)
        targets = np.load(target_file)
        
        cm = confusion_matrix(targets, preds, labels=class_labels)
        
        # Normalization by line (number of real samples in each class)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            # Converts Nan to 0 (in case a class does not appear in fold)
            cm_normalized = np.nan_to_num(cm_normalized)
        
        normalized_cms.append(cm_normalized)

    if not normalized_cms:
        print("No confusion matrix has been calculated.")
        return

    # Calculates mean and std between folds
    mean_cm = np.mean(normalized_cms, axis=0)
    std_cm = np.std(normalized_cms, axis=0)

    annotations = np.empty_like(mean_cm, dtype=object)
    for i in range(mean_cm.shape[0]):
        for j in range(mean_cm.shape[1]):
            mean_val = mean_cm[i, j]
            std_val = std_cm[i, j]
            annotations[i, j] = f"{mean_val:.1%} ± {std_val:.1%}"

    # Plots heatmap
    class_names = ['Class A', 'Class B', 'Class C', 'Class D']
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(mean_cm, annot=annotations, fmt="", cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 12})
    
    plt.title(f'Confusion Matrix (Mean ± Desvio Padrão) between {len(pred_files)} Folds\nModel: {results_path.parent.name} - HPs: {results_path.name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Salva a figura
    save_path = results_path / "confusion_matrix_mean_std.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"\nConfusion matrix saved in: {save_path}")
    print("\Mean (Normalized):")
    np.set_printoptions(precision=4)
    print(mean_cm)
    print("\nStd (Normalized):")
    print(std_cm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generates the confusion matrix (mean ± std) for a trained model.')
    parser.add_argument('results_path', type=str, 
                        help='Path for results directory of a specific model. e.g.: ./results/production/MLP/hidden_256_128_dropout_0.2_lr_0.001')
    
    args = parser.parse_args()
    
    results_path = Path(args.results_path)
    if not results_path.is_dir():
        print(f"Error: Specified directory does not exists: {results_path}")
    else:
        plot_confusion_matrix(results_path)