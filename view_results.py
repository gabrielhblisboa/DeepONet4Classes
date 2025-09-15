import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import argparse
from pathlib import Path
import glob
import csv
from datetime import datetime

def plot_and_save_metrics(results_path: Path):
    """
    Calculates performance metrics, plots confusion matrices, and appends the
    final metrics to a central CSV file in percentage format 'mean% ± std%'.
    """
    pred_files = sorted(glob.glob(str(results_path / "data" / "predictions_fold_*.npy")))
    target_files = sorted(glob.glob(str(results_path / "data" / "targets_fold_*.npy")))

    if not pred_files or not target_files:
        print(f"Erro: Predict/target files not found in {results_path / 'data'}")
        return
    
    # --- Coleta de métricas por fold (código inalterado) ---
    fold_accuracies = []
    macro_precisions, macro_recalls, macro_f1s = [], [], []
    weighted_precisions, weighted_recalls, weighted_f1s = [], [], []
    
    for fold_idx, (pred_file, target_file) in enumerate(zip(pred_files, target_files), 1):
        print(f"--- Processing Fold {fold_idx} ---")
        preds = np.load(pred_file)
        targets = np.load(target_file)
        
        fold_accuracies.append(accuracy_score(preds, targets))
        
        report = classification_report(targets, preds, output_dict=True, zero_division=0)
        
        macro_precisions.append(report['macro avg']['precision'])
        macro_recalls.append(report['macro avg']['recall'])
        macro_f1s.append(report['macro avg']['f1-score'])
        weighted_precisions.append(report['weighted avg']['precision'])
        weighted_recalls.append(report['weighted avg']['recall'])
        weighted_f1s.append(report['weighted avg']['f1-score'])
        
    # --- Seção de prints no terminal (inalterada) ---

    # --- SEÇÃO MODIFICADA: Salvar as métricas em formato de porcentagem ---
    if fold_accuracies:
        csv_path = results_path.parent.parent / "model_comparison.csv"
        
        # Prepara o dicionário com os dados no formato "XX.XX% ± YY.YY%"
        results_dict = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': results_path.parent.name,
            'hyperparameters': results_path.name,
            'accuracy': f"{np.mean(fold_accuracies):.2%} ± {np.std(fold_accuracies):.2%}",
            'weighted_precision': f"{np.mean(weighted_precisions):.2%} ± {np.std(weighted_precisions):.2%}",
            'weighted_recall': f"{np.mean(weighted_recalls):.2%} ± {np.std(weighted_recalls):.2%}",
            'weighted_f1': f"{np.mean(weighted_f1s):.2%} ± {np.std(weighted_f1s):.2%}",
            'macro_f1': f"{np.mean(macro_f1s):.2%} ± {np.std(macro_f1s):.2%}",
        }

        fieldnames = list(results_dict.keys())
        file_exists = csv_path.is_file()

        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(results_dict)
        
        print(f"\n--- Resultados salvos com sucesso em: {csv_path} ---")
    # --- FIM DA SEÇÃO MODIFICADA ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generates performance metrics and appends them to a summary CSV.')
    parser.add_argument('results_path', type=str, 
                        help='Path for results directory. e.g.: ./results/production/MLP/hidden_256_128')
    args = parser.parse_args()
    results_path = Path(args.results_path)
    if not results_path.is_dir():
        print(f"Error: Specified directory does not exists: {results_path}")
    else:
        plot_and_save_metrics(results_path)