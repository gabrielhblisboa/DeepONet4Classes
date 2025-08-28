import torch
# Adicionando MLP e Trainer aos imports
from src.models.mlp import MLP
from src.models.deeponet import DeepONet
from src.models.multistask import ConvAutoencoderMultitask, MultitaskAutoencoder, MultitaskUNet
from src.training import Trainer, DeepONetTrainer, MultitaskTrainer, calculate_class_weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
from src.data_handling import CustomDataloader, DeepONetDataloader, LoroCV
from src.visualization import plot_lofargram, plot_tsne_embeddings, palette
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
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

def build_sub_net(net_config, input_size, p_dim):
    """Função auxiliar para construir as redes da Branch e Trunk."""
    if net_config['name'] == 'MLP':
        # Adiciona uma camada de saída para garantir a dimensão p_dim
        hidden_channels = net_config['params']['hidden_channels']
        model = MLP(input_shape=input_size, 
                    hidden_channels=hidden_channels + [p_dim], 
                    n_targets=None, # A última camada já é a de p_dim
                    activation_output_layer=None, # Sem ativação final
                    dropout=net_config['params']['dropout'])
    # Futuramente, você pode adicionar outras arquiteturas aqui (ex: 'CNN')
    # elif net_config['name'] == 'CNN':
    #     ...
    else:
        raise ValueError(f"Sub-network {net_config['name']} não reconhecida.")
    return model

def model_select(config, branch_net = None):
    window_size = config.window_size
    
    if config.model_name == "MultitaskAutoencoder":
        latent_dim_size = config.latent_dim_size
        output_size = config.output_size
        return lambda input_size: MultitaskAutoencoder(input_size, [latent_dim_size, 32], output_size)
    
    elif config.model_name == "ConvAutoencoderMultitask":
        latent_dim_size = config.latent_dim_size
        return lambda input_size: ConvAutoencoderMultitask(window_size, 512, 4, latent_dim_size=latent_dim_size, dropout_rate=0.5)
    
    elif config.model_name == "MultitaskUNet":
        latent_dim_size = config.latent_dim_size
        return lambda input_size: MultitaskUNet(window_size, 512, 4, latent_dim_size=latent_dim_size)
    
    elif config.model_name == "MLP":
        return lambda input_size: MLP(input_shape=input_size, hidden_channels=config.hidden_channels, n_targets=4, dropout=config.dropout)
    
    # elif config.model_name == "DeepONet-MLP":
    #     return lambda input_size: DeepONet(branch_net= MLP(input_shape=input_size,
    #                                                        hidden_channels=config.hidden_channels, 
    #                                                        n_targets=config.embedding_dim, 
    #                                                        dropout=config.dropout),
    #                                        n_targets=config.embedding_dim, 
    #                                        embedding_dim=config.embedding_dim)
    
    elif config.model_name == "DeepONet":
        # A função de construção do modelo agora é mais complexa
        def model_builder_func(input_size):
            # Constrói a Branch Net
            branch_net_config = config.branch_net
            branch_input_size = input_size
            branch_net = build_sub_net(branch_net_config, branch_input_size, config.p_dim)
            
            # Constrói a Trunk Net
            trunk_net_config = config.trunk_net
            trunk_input_size = 4 # One-hot para 4 classes
            trunk_net = build_sub_net(trunk_net_config, trunk_input_size, config.p_dim)

            return DeepONet(branch_net=branch_net, trunk_net=trunk_net, p_dim=config.p_dim)
        return model_builder_func
    
    else:
        raise ValueError(f"Model name {config.model_name} not recognized.")

def run_experiment(config, lofar_data, results_path, device):
    alpha = config.alpha if hasattr(config, 'alpha') else None
    window_size = config.window_size
    non_multitask_models_list = ["MLP", "DeepONet"] # Adicionado DeepONet

    overlap = None
    if window_size == 16: overlap = 14
    elif window_size == 32: overlap = 28
    
    model_builder = model_select(config)
    lorocv = LoroCV(n_splits=5, window_size=window_size, overlap=overlap, random_seed=42)

    fold = config.fold
    for i, (X_train, y_train, X_test, y_test) in enumerate(lorocv.split(lofar_data)):
        if i != fold:
            continue
        
        # --- NOVO BLOCO DE LÓGICA PARA DEEPONET ---
        if config.model_name == "DeepONet":
            if config.branch_net['name'] == 'MLP' and len(X_train.shape) > 2:
                 X_train = X_train.reshape(X_train.shape[0], -1)
                 X_test = X_test.reshape(X_test.shape[0], -1)
            
            X_train_t = torch.from_numpy(X_train).float().to(device)
            y_train_t = torch.from_numpy(y_train).long().to(device)
            X_test_t = torch.from_numpy(X_test).float().to(device)
            y_test_t = torch.from_numpy(y_test).long().to(device)

            train_dataset_fold = DeepONetDataloader(X_train_t, y_train_t)
            test_dataset_fold = DeepONetDataloader(X_test_t, y_test_t)
            train_loader_fold = DataLoader(train_dataset_fold, batch_size=32, shuffle=True)
            test_loader_fold = DataLoader(test_dataset_fold, batch_size=32, shuffle=False)

            input_size = X_train.shape[-1]
            model_fold = model_builder(input_size).to(device)
            optimizer_fold = torch.optim.Adam(model_fold.parameters(), lr=config.learning_rate)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_fold, gamma=0.93)
            criterion = torch.nn.BCEWithLogitsLoss()

            trainer_fold = DeepONetTrainer(model_fold, optimizer_fold, scheduler, criterion, num_epochs=100, verbose=True, wandb_logging=True)
            trainer_fold.train(train_loader_fold, test_loader_fold, patience=10)

            y_pred, y_target = trainer_fold.evaluate_final(X_test_t, y_test_t)
            
            accuracy = accuracy_score(y_target, y_pred)
            precision = precision_score(y_target, y_pred, average='weighted', zero_division=0)
            recall = np.mean(recall_score(y_target, y_pred, average=None, zero_division=0))
            f1 = f1_score(y_target, y_pred, average='weighted', zero_division=0)
            wandb.log({"final_accuracy": accuracy, "final_precision": precision, "final_recall": recall, "final_f1_score": f1})
            
            (results_path / "data").mkdir(parents=True, exist_ok=True)
            np.save(results_path / "data" / f"predictions_fold_{fold}.npy", y_pred)
            np.save(results_path / "data" / f"targets_fold_{fold}.npy", y_target)
            continue
        
        # --- LÓGICA PARA MLP E MODELOS MULTITASK (A SUA ESTRUTURA ORIGINAL) ---
        class_weights = calculate_class_weights(y_train).to(device)
        is2d = window_size is not None and config.model_name != "MLP"
        train_dataset_fold = CustomDataloader(X_train, y_train, is2d=is2d, device=device)
        test_dataset_fold = CustomDataloader(X_test, y_test, is2d=is2d, device=device)
        train_loader_fold = DataLoader(train_dataset_fold, batch_size=32, shuffle=True)
        test_loader_fold = DataLoader(test_dataset_fold, batch_size=32, shuffle=False)

        input_size = X_train.shape[1] if not is2d else (X_train.shape[1], X_train.shape[2])
        model_fold = model_builder(input_size).to(device)
        optimizer_fold = torch.optim.Adam(model_fold.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_fold, gamma=0.93)
        clf_criterion_fold = torch.nn.CrossEntropyLoss(weight=class_weights)

        if config.model_name == "MLP":
            trainer_fold = Trainer(model_fold, optimizer_fold, scheduler, clf_criterion_fold, num_epochs=100, verbose=True, wandb_logging=True)
            trainer_fold.train(train_loader_fold, test_loader_fold, patience=10)
            _, accuracy, precision, recall, f1, roc_auc, y_pred, y_target = trainer_fold.evaluate(test_loader_fold)
            wandb.log({"final_accuracy": accuracy, "final_precision": precision, "final_recall": np.mean(recall), "final_f1_score": f1, "final_roc_auc": roc_auc})
            
            (results_path / "data").mkdir(parents=True, exist_ok=True)
            np.save(results_path / "data" / f"predictions_fold_{fold}.npy", y_pred)
            np.save(results_path / "data" / f"targets_fold_{fold}.npy", y_target)

        else: # Modelos Multitask
            rec_criterion_fold = torch.nn.MSELoss()
            trainer_fold = MultitaskTrainer(model_fold, optimizer_fold, scheduler, clf_criterion_fold, rec_criterion_fold, alpha=alpha, num_epochs=100, verbose=True, wandb_logging=True)
            trainer_fold.train(train_loader_fold, test_loader_fold, patience=10)
            
            val_loss, accuracy, precision, recall, f1, sp, roc_auc, y_pred, y_target = trainer_fold.evaluate(test_loader_fold)
            wandb.log({"final_accuracy": accuracy, "final_precision": precision, "final_recall": np.mean(recall), "final_f1_score": f1, "final_sp_index": sp, "final_roc_auc": roc_auc})

            (results_path / "data").mkdir(parents=True, exist_ok=True)
            np.save(results_path / "data" / f"predictions_fold_{fold}.npy", y_pred)
            np.save(results_path / "data" / f"targets_fold_{fold}.npy", y_target)

            fold_embeddings, fold_targets, fold_scores = trainer_fold.evaluate_embeddings(test_loader_fold)
            save_embeddings_and_targets(i, fold_embeddings, fold_targets, results_path)
            wandb.log(fold_scores)
            
            # ... (Lógica de plots do t-SNE e continuity) ...
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
                distances_from_start = ce_scores["scores"]["spearman"]["distances_from_start"]
                time_ranks = ce_scores["scores"]["spearman"]["time_ranks"]
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
                axf[i_cls].scatter(cls_embeddings[:, 0], cls_embeddings[:, 1], s=1, c=indices, cmap='inferno', alpha=0.7)
                axf[i_cls].set_xlabel('t-SNE Dimension 1', fontsize=8)
                axf[i_cls].set_ylabel('t-SNE Dimension 2', fontsize=8)
                axf[i_cls].grid(True, linestyle='--', alpha=0.5)
                axf[i_cls].axhline(0, color='black', linewidth=0.5)
                axf[i_cls].axvline(0, color='black', linewidth=0.5)
            norm = plt.Normalize(vmin=0, vmax=1)
            sm = plt.cm.ScalarMappable(cmap='inferno', norm=norm)
            cbar = fig.colorbar(sm, ax=axf, orientation='vertical', pad=0.1, fraction=0.02)
            cbar.set_label('Normalized Time Progression')
            plot_name = f"t-SNE_embeddings-CONTINUITY_fold_{i}"
            fig.savefig(results_path / "plots" / "png" / f"{plot_name}.png", bbox_inches='tight', dpi=300)
            fig.savefig(results_path / "plots" / "svg" / f"{plot_name}.svg", bbox_inches='tight')
            wandb.log({"t-SNE continuity plot": wandb.Image(fig)})
            plt.close(fig)



def make_hp_name(config):
    alpha = config.alpha if hasattr(config, 'alpha') else 'na'
    latent_dim_size = config.latent_dim_size if hasattr(config, 'latent_dim_size') else 'na'
    output_size = config.output_size if hasattr(config, 'output_size') else 'na'
    window_size = config.window_size
    learning_rate = config.learning_rate
    

    if config.model_name == "MLP":
        hidden_str = '_'.join(map(str, config.hidden_channels))
        return f"hidden_{hidden_str}_dropout_{config.dropout}_lr_{learning_rate}"
    if config.model_name == "DeepONet-MLP":
         hidden_str = '_'.join(map(str, config.hidden_channels))
         return f"hidden_{hidden_str}_dropout_{config.dropout}_lr_{learning_rate}_embedding_{config.embedding_dim}"
    elif config.model_name == "MultitaskAutoencoder":
        return f"alpha_{alpha}_latent_{latent_dim_size}_window_{window_size}_lr_{learning_rate}"
    elif config.model_name == "ConvAutoencoderMultitask":
        return f"alpha_{alpha}_latent_{latent_dim_size}_output_{output_size}_window_{window_size}_lr_{learning_rate}"
    elif config.model_name == "MultitaskUNet":
        return f"alpha_{alpha}_latent_{latent_dim_size}_output_{output_size}_window_{window_size}_lr_{learning_rate}"
    else:
        raise ValueError(f"Model name {config.model_name} not recognized.")

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
    wandb.init(project=project_name)
    config = wandb.config

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    if args.debug:
        results_path = Path(f"./results/debug/{config.model_name}/{hp_name}")
    else:
        results_path = Path(f"./results/production/{config.model_name}/{hp_name}")

    (results_path / "plots" / "svg").mkdir(parents=True, exist_ok=True)
    (results_path / "plots" / "png").mkdir(parents=True, exist_ok=True)
    (results_path / "data").mkdir(parents=True, exist_ok=True)

    lofar_data, _, _ = load_data()

    embeddings, all_targets = run_experiment(config, lofar_data, results_path, device)

    if config.model_name != "MLP":
        save_embeddings_and_targets(config, embeddings, all_targets, results_path)

    store_hash(model_hash)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the model.')
    parser.add_argument('--config', type=str, default='config', help='Path to the configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    # lofar_data, _, _ = load_data()

    config_file = f"./configs/{args.config}.json"
    with open(config_file, 'r') as f:
        sweep_configuration = json.load(f)

    if args.debug:
        project_name = f'DeepONet-debug-v1'
    else:
        project_name = f'DeepONet-v1'
    sweep_configuration['name'] = f"{project_name}-sweep"

    sweep_id = wandb.sweep(sweep_configuration, project=project_name)

    wandb.agent(sweep_id, function=lambda : sweep_experiment(project_name))
