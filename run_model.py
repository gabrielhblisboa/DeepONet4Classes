import torch
from src.model import ConvAutoencoderMultitask, MultitaskAutoencoder, MultitaskUNet
from src.io.offline import load_raw_data
from src.signal.passivesonar import lofar
from src.signal.utils import resample
from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import argparse
import wandb
from src.data_handling import CustomDataloader, LoroCV
from src.visualization import plot_lofargram, plot_tsne_embeddings, palette
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from src.training import MultitaskTrainer, calculate_class_weights
import matplotlib
matplotlib.use('agg')

def save_embeddings_and_targets(config, embeddings, all_targets, results_path):
    """
    Save each embedding and target array into separate .npy files.

    Parameters:
        embeddings (list of arrays): List of embedding arrays from each fold.
        all_targets (list of arrays): List of target arrays from each fold.
        results_path (Path): Path object to the directory where files will be saved.
    """
    for i, (embedding, target) in enumerate(zip(embeddings, all_targets)):
        embedding = np.array(embedding)
        target = np.array(target)

        np.save(results_path / "data" / f"embeddings_fold_{config.fold}.npy", embedding)
        np.save(results_path / "data" / f"all_targets_fold_{config.fold}.npy", target)

def load_data():
    datapath = Path("../data/4classes/") 
    raw_data = load_raw_data(datapath)

    n_pts_fft = 1024
    n_overlap = 0
    decimation_rate = 3
    final_fs = 22050//decimation_rate

    raw_data = raw_data.apply(lambda rr: resample(rr['signal'], rr['fs'], final_fs = final_fs))
    lofar_data = raw_data.apply(lofar,
                                fs=final_fs,
                                n_pts_fft = n_pts_fft, 
                                n_overlap=n_overlap, 
                                spectrum_bins_left=512)

    class_map = {
        "ClassA": 0,
        "ClassB": 1,
        "ClassC": 2,
        "ClassD": 3,
    }

    trgt = np.concatenate([class_map[cls_name]*np.ones(Sxx.shape[0])
                    for cls_name, run in lofar_data.items() 
                    for run_name, (Sxx, _, _) in run.items()])

    data = np.concatenate([ Sxx
                    for cls_name, run in lofar_data.items() 
                    for run_name, (Sxx, _, _) in run.items()], axis=0)

    print("=" * 75)
    print("Completed Data Preprocessing with the Following Configuration:")
    print(f" - FFT Points               : {n_pts_fft}")
    print(f" - Window Overlap           : {n_overlap}")
    print(f" - Decimation Rate          : {decimation_rate}")
    print(f" - Final Sampling Frequency : {final_fs}")
    print()
    print("Data Shapes:")
    print(f" - Input Data Shape         : {data.shape}")
    print(f" - Target Data Shape        : {trgt.shape}")
    print("=" * 75)

    return lofar_data, data, trgt

def model_select(config):
    latent_dim_size = config.latent_dim_size
    output_size = config.output_size
    window_size = config.window_size
    if config.model_name == "MultitaskAutoencoder":
        return lambda input_size: MultitaskAutoencoder(input_size, [latent_dim_size, 32], output_size)
    elif config.model_name == "ConvAutoencoderMultitask":
        return lambda input_size: ConvAutoencoderMultitask(window_size, 512, 4, latent_dim_size=latent_dim_size, dropout_rate=0.5)
    elif config.model_name == "MultitaskUNet":
        return lambda input_size: MultitaskUNet(window_size, 512, 4, latent_dim_size=latent_dim_size)
    else:
        raise ValueError(f"Model name {config.model_name} not recognized.")

def run_experiment(config, lofar_data, results_path, device):
    # Initialize the model, optimizer, and criterion
    # latent_dim_size = config.latent_dim_size
    # output_size = config.output_size
    alpha = config.alpha
    window_size = config.window_size

    if window_size is None:
        overlap = None
    elif window_size == 16:
        overlap = 14
    elif window_size == 32:
        overlap = 28
    else:
        raise ValueError(f"Window size {window_size} not recognized.")
    
    model_builder = model_select(config)
    # Perform cross-validation using LoroCV
    accuracies = []
    embeddings = []
    all_targets    = []
    lorocv_no_window = LoroCV(n_splits=5, window_size=window_size, overlap=overlap, random_seed=42)

    fold = config.fold
    for i, (X_train, y_train, X_test, y_test) in enumerate(lorocv_no_window.split(lofar_data)):
        if i != fold:
            continue
        # Compute class weights for loss balancing
        class_weights = calculate_class_weights(y_train).to(device)
        # Standardize the data
        # scaler = StandardScaler()
        # shape = X_train.shape
        # X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(shape)
        # shape = X_test.shape
        # X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(shape)

        # Create DataLoader instances for the fold
        is2d = window_size is not None
        train_dataset_fold = CustomDataloader(X_train, y_train, is2d=is2d, device=device)
        test_dataset_fold = CustomDataloader(X_test, y_test, is2d=is2d, device=device)
        train_loader_fold = DataLoader(train_dataset_fold, batch_size=32, shuffle=True)
        test_loader_fold = DataLoader(test_dataset_fold, batch_size=32, shuffle=False)

        # Initialize the model, optimizer, and criterion for the fold
        input_size = X_train.shape[1]
        model_fold = model_builder(input_size).to(device)
        optimizer_fold = torch.optim.Adam(model_fold.parameters(), lr=config.learning_rate)
        # Exponential Decay Learning Rate Scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_fold, gamma=0.93)  # Adjust gamma to your needs
        # scheduler = None

        clf_criterion_fold = torch.nn.CrossEntropyLoss(weight=class_weights)
        rec_criterion_fold = torch.nn.MSELoss()

        # Initialize and run the trainer for the fold
        trainer_fold = MultitaskTrainer(model_fold, optimizer_fold, scheduler,
                                        clf_criterion_fold, 
                                        rec_criterion_fold, 
                                        alpha=alpha,
                                        num_epochs=100, verbose=True, wandb_logging=True)
        trainer_fold.train(train_loader_fold, test_loader_fold, patience=10)

        # Evaluate the model on the test fold
        _, accuracy, precision, recall, f1, sp, roc_auc = trainer_fold.evaluate(test_loader_fold)
        wandb.log({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "sp_index": sp,
            "roc_auc": roc_auc
        })

        # Evaluate the embeddings
        fold_embeddings, fold_targets, fold_scores = trainer_fold.evaluate_embeddings(test_loader_fold)
        embeddings.append(fold_embeddings)
        all_targets.append(fold_targets)

        accuracies.append(accuracy)

        wandb.log(fold_scores)

        fig, ax = plt.subplots(figsize=(12, 12/ 1.618))
        plot_tsne_embeddings(ax, fold_embeddings, fold_targets, palette=palette)
        plot_name = f"t-SNE_embeddings_fold_{i}"
        fig.savefig(results_path / "plots" / "png" / f"{plot_name}.png", bbox_inches='tight', dpi=300)
        fig.savefig(results_path / "plots" / "svg" / f"{plot_name}.svg", bbox_inches='tight')

        wandb.log({"t-SNE plot": wandb.Image(fig)})
        plt.close(fig)

        continuous_eval = trainer_fold.evaluate_embeddings_continuity(fold_embeddings, fold_targets)
        continuous_embeddings = {i_cls: continuous_eval[i_cls]["class_embeddings"] for i_cls in continuous_eval.keys()}

        fig, axes = plt.subplots(4, 2, figsize=(8, 8/ 1.618))
        for i_cls, ce_scores in continuous_eval.items():
            spearman_correlation = ce_scores["scores"]["spearman"]["correlation"]
            distances_from_start = ce_scores["scores"]["spearman"]["distances_from_start"]
            time_ranks = ce_scores["scores"]["spearman"]["time_ranks"]


            local_consistency = ce_scores["scores"]["local_consistency"]["local_consistency"]
            step_lengths = ce_scores["scores"]["local_consistency"]["step_lengths"]

            axes[i_cls, 0].plot(time_ranks, distances_from_start)
            axes[i_cls, 1].plot(time_ranks, step_lengths)

        plot_name = f"continuity_measures_fold{i}"
        fig.savefig(results_path / "plots" / "png" / f"{plot_name}.png", bbox_inches='tight', dpi=300)
        fig.savefig(results_path / "plots" / "svg" / f"{plot_name}.svg", bbox_inches='tight')

        wandb.log({"Continuity Measures": wandb.Image(fig)})
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(8, 8/ 1.618), sharex=True, sharey=True)
        axf = axes.flat
        for i_cls, (trgt, cls_embeddings) in enumerate(continuous_embeddings.items()):
            num_windows = len(cls_embeddings)
            indices = np.array(list(range(num_windows)))/num_windows
            # Plot the connecting lines
            axf[i_cls].scatter(cls_embeddings[:, 0], cls_embeddings[:, 1], s=1,
                           c=indices, cmap='inferno', alpha=0.7)

            # arrow_indices = [int(num_windows * 0.2), int(num_windows * 0.5), int(num_windows * 0.8)]
            # for idx in arrow_indices:
            #     p1 = cls_embeddings[idx]
            #     # Make sure there is a next point to draw to
            #     if idx + 1 < num_windows:
            #         p2 = cls_embeddings[idx + 1]
            #         # Calculate direction vector and draw arrow
            #         dx = p2[0] - p1[0]
            #         dy = p2[1] - p1[1]
            #         ax.arrow(p1[0], p1[1], dx, dy,
            #                 head_width=0.5, head_length=0.5, fc=palette[i], ec=palette[i],
            #                 length_includes_head=True, zorder=3)

            # axf[i].set_title('Evolution of Multiple Time Series in t-SNE Space', fontsize=18)
            axf[i_cls].set_xlabel('t-SNE Dimension 1', fontsize=8)
            axf[i_cls].set_ylabel('t-SNE Dimension 2', fontsize=8)
            # axf[i].legend(fontsize=12)
            axf[i_cls].grid(True, linestyle='--', alpha=0.5)
            axf[i_cls].axhline(0, color='black', linewidth=0.5)
            axf[i_cls].axvline(0, color='black', linewidth=0.5)
            # ax.set_aspect('equal', adjustable='box') # Often helpful for t-SNE

        # Create a ScalarMappable object that understands the 'inferno' colormap and data range (0 to 1)
        norm = plt.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap='inferno', norm=norm)

        # Add the colorbar to the figure, positioning it at the bottom.
        # 'ax=axf' tells it to steal space from all the subplots.
        # 'orientation' makes it horizontal. 'pad' and 'fraction' control spacing.
        cbar = fig.colorbar(sm, ax=axf, orientation='vertical', pad=0.1, fraction=0.02)
        cbar.set_label('Normalized Time Progression')


        plot_name = f"t-SNE_embeddings-CONTINUITY_fold_{i}"
        fig.savefig(results_path / "plots" / "png" / f"{plot_name}.png", bbox_inches='tight', dpi=300)
        fig.savefig(results_path / "plots" / "svg" / f"{plot_name}.svg", bbox_inches='tight')

        wandb.log({"t-SNE continuity plot": wandb.Image(fig)})
        plt.close(fig)

    return embeddings, all_targets

def make_hp_name(config):
    alpha = config.alpha
    latent_dim_size = config.latent_dim_size
    output_size = config.output_size
    window_size = config.window_size
    learning_rate = config.learning_rate

    if config.model_name == "MultitaskAutoencoder":
        return f"alpha_{alpha}_latent_{latent_dim_size}_window_{window_size}_lr_{learning_rate}"
    elif config.model_name == "ConvAutoencoderMultitask":
        return f"alpha_{alpha}_latent_{latent_dim_size}_output_{output_size}_window_{window_size}_lr_{learning_rate}"
    elif config.model_name == "MultitaskUNet":
        return f"alpha_{alpha}_latent_{latent_dim_size}_output_{output_size}_window_{window_size}_lr_{learning_rate}"
    else:
        raise ValueError(f"Model name {config.model_name} not recognized.")

# class Config:
#     def __init__(self, config_file):
#         with open(config_file, 'r') as f:
#             config_data = json.load(f)
        
#         self.model_name = config_data.get('model_name', 'default_model_name')
#         self.hidden_layer_sizes = config_data.get('hidden_layer_sizes', [128, 64, 32])
#         self.output_size = config_data.get('output_size', 4)
#         self.alpha = config_data.get('alpha', 0.5)
#         self.window_size = config_data.get('window_size', None)

def has_been_run(hash):
    hash_file = "config_hashes.txt"
    if not os.path.exists(hash_file):
        return False
    with open(hash_file, "r") as file:
        existing_hashes = file.read().split()
    return hash in existing_hashes

def store_hash(hash):
    with open("config_hashes.txt", "a") as file:
        file.write(hash + "\n")

def sweep_experiment(project_name):
    # Run the experiment with the provided config
    # rng_key = random.PRNGKey(42)
    
    # Initialize wandb
    wandb.init(project=project_name)
    config = wandb.config

    # ======================================================
    # CONFIGURATION
    # ======================================================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load configuration from the specified config file
    # config = Config(Path(f"./configs/{args.config}.json"))
    model_name = config.model_name
    hp_name = make_hp_name(config)
    fold = config.fold

    model_id = f"{model_name}_{hp_name}"
    model_hash = f"Fold_{fold}_{model_id}"

    if has_been_run(model_hash):
        print("Configuration has already been run. Skipping...")
        wandb.log({"duplicate": True})
        return
    config.model_id = model_id

    # ======================================================

    # ======================================================

    if args.debug:
        results_path = Path(f"./results/debug/{config.model_name}/{hp_name}")
    else:
        results_path = Path(f"./results/production/{config.model_name}/{hp_name}")
    
    (results_path / "plots" / "svg").mkdir(parents=True, exist_ok=True)
    (results_path / "plots" / "png").mkdir(parents=True, exist_ok=True)
    (results_path / "data").mkdir(parents=True, exist_ok=True)

    # ======================================================
    # DATA PROCESSING
    # ======================================================

    lofar_data, _, _ = load_data()

    # ======================================================
    # MODEL TRAINING AND EVALUATION
    # ======================================================

    embeddings, all_targets = run_experiment(config, lofar_data, results_path, device)

    save_embeddings_and_targets(config, embeddings, all_targets, results_path)

    store_hash(model_hash)



if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the model.')
    parser.add_argument('--config', type=str, default='config', help='Path to the configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    # Load data
    lofar_data, _, _ = load_data()

    
    # ======================================================
    # WANDB SWEEP CONFIGURATION
    # ======================================================
    config_file = f"./configs/{args.config}.json"
    with open(config_file, 'r') as f:
        sweep_configuration = json.load(f)
        # sweep_configuration = json.dumps(sweep_configuration)
        
    if args.debug:
        project_name = f'DeepLearning-debug'
    else:
        project_name = f'DeepLearning-v4'
    sweep_configuration['name'] = f"{project_name}-sweep"

    sweep_id = wandb.sweep(sweep_configuration, project=project_name)

    wandb.agent(sweep_id, function=lambda : sweep_experiment(project_name))
