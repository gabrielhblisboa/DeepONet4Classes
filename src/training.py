
import copy
from sklearn.utils.class_weight import compute_class_weight
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from alive_progress import alive_bar
import numpy as np
from sklearn.manifold import TSNE
import wandb
import skdim
from sklearn.manifold import trustworthiness
from skdim.id import KNN
from scipy.stats import spearmanr
from scipy.spatial.distance import euclidean, cosine


def calculate_local_step_consistency(series_embeddings: np.ndarray) -> float:
    """
    Measures the smoothness of the trajectory's velocity via Coefficient of Variation.
    A lower score is better (less jerky).
    """
    if len(series_embeddings) < 2:
        return np.nan
        
    # Calculate the length of each step between consecutive points
    step_lengths = [cosine(series_embeddings[i], series_embeddings[i+1]) 
                    for i in range(len(series_embeddings) - 1)]
    
    if not step_lengths or np.mean(step_lengths) == 0:
        return np.nan

    # Calculate Coefficient of Variation (CV) = std / mean
    mean_step = np.mean(step_lengths)
    std_step = np.std(step_lengths)
    
    return std_step / mean_step, step_lengths

def calculate_spearman_rank_correlation(series_embeddings: np.ndarray) -> float:
    """
    Measures if the trajectory is smoothly unfurling from its start point.
    A score close to +1.0 is best.
    """
    if len(series_embeddings) < 3:
        return np.nan # Cannot compute correlation with less than 3 points
        
    # Take the first point as the reference
    start_point = series_embeddings[0]
    
    # Calculate Euclidean distance from the start to all other points
    distances_from_start = [cosine(start_point, point) for point in series_embeddings[1:]]
    
    # The time ranks are just the order of the points
    time_ranks = np.arange(1, len(series_embeddings))
    
    # Calculate Spearman's rank correlation
    correlation, p_value = spearmanr(time_ranks, distances_from_start)
    
    return correlation, distances_from_start, time_ranks


def calculate_class_weights(y_train):
    # Compute class weights using sklearn's compute_class_weight
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    # Convert the class weights to a PyTorch tensor
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    
    return class_weights_tensor

def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

def sp_index(recall):
    return np.sqrt(recall.mean() * geo_mean(recall))


class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, num_epochs=10, verbose=False, plotpath=None, wandb_logging=False):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.plotpath = plotpath
        self.wandb_logging = wandb_logging

    def train(self, train_loader, test_loader, patience=10):
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            with alive_bar(len(train_loader), title=f"Training Epoch {epoch+1}/{self.num_epochs}") as bar:
                for batch_data, batch_target in train_loader:
                    self.optimizer.zero_grad()
                    output = self.model(batch_data)
                    loss = self.criterion(output, batch_target)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    bar()
            
            val_loss, accuracy, precision, recall, f1, roc_auc, y_pred, y_target = self.evaluate(test_loader)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("=" * 80)
                    print(f"Early stopping at epoch {epoch+1}. Restoring best model state.")
                    print("=" * 80)
                    break
            
            if self.scheduler is not None:
                self.scheduler.step()
                lr = self.scheduler.get_last_lr()[0]
            else:
                lr = None
            
            if self.wandb_logging:
                wandb.log({
                    'epoch': epoch + 1,
                    'loss': epoch_loss / len(train_loader),
                    'val_loss': val_loss,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': np.mean(recall),
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'learning_rate': lr
                })

            if self.verbose:
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}, "
                      f"Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                      f"Recall: {np.mean(recall):.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self.model

    def evaluate(self, test_loader):
        self.model.eval()
        all_preds = []
        all_targets = []
        val_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_target in test_loader:
                output = self.model(batch_data)
                loss = self.criterion(output, batch_target)
                val_loss += loss.item()
                _, preds = torch.max(output, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_target.cpu().numpy())
        
        val_loss /= len(test_loader)
        y_pred = np.array(all_preds)
        y_target = np.array(all_targets)
        
        accuracy = np.mean(recall_score(y_target, y_pred, average=None))
        precision = precision_score(y_target, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_target, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_target, y_pred, average='weighted', zero_division=0)
        roc_auc = roc_auc_score(
            label_binarize(y_target, classes=[0, 1, 2, 3]),
            label_binarize(y_pred, classes=[0, 1, 2, 3]),
            average='weighted',
            multi_class='ovr'
        )
        return val_loss, accuracy, precision, recall, f1, roc_auc, y_pred, y_target
        

class MultitaskTrainer(Trainer):
    def __init__(
        self, 
        model, 
        optimizer, 
        scheduler,
        classification_criterion, 
        reconstruction_criterion, 
        alpha=0.0, 
        num_epochs=10, 
        verbose=False, 
        plotpath=None,
        wandb_logging=False
    ):
        super(MultitaskTrainer, self).__init__(model, optimizer, scheduler, classification_criterion, num_epochs, verbose, plotpath)
        self.reconstruction_criterion = reconstruction_criterion
        self.alpha = alpha
        self.wanbd_logging = wandb_logging

    def train(self, train_loader, test_loader, patience):
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            rec_epoch_loss = 0.0
            clf_epoch_loss = 0.0
            with alive_bar(len(train_loader), title=f"Training Epoch {epoch+1}/{self.num_epochs}") as bar:
                for batch_data, batch_target in train_loader:
                    self.optimizer.zero_grad()
                    class_output, reconstructed = self.model(batch_data)
                    classification_loss = self.criterion(class_output, batch_target)
                    reconstruction_loss = self.reconstruction_criterion(reconstructed, batch_data)

                    if self.alpha > 0:
                        total_loss = self.alpha * reconstruction_loss + (1 - self.alpha) * classification_loss
                    elif self.alpha == 0:
                        total_loss = classification_loss
                    else:
                        raise ValueError("Alpha must be between 0 and 1.")

                    total_loss.backward()
                    self.optimizer.step()
                    epoch_loss += total_loss.item()
                    rec_epoch_loss += reconstruction_loss.item()
                    clf_epoch_loss += classification_loss.item()
                    bar()
            
            if self.scheduler is not None:
                self.scheduler.step()
            val_loss, accuracy, precision, recall, f1, sp, roc_auc = self.evaluate(test_loader)

            if val_loss['total_loss'] < best_loss:
                best_loss = copy.deepcopy(val_loss['total_loss'])
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("=" * 80)
                    print(f"Early stopping at epoch {epoch+1}. Restoring best model state.")
                    print("=" * 80)
                    break
            if self.scheduler is not None:
                lr = self.scheduler.get_last_lr()[0]
            else:
                lr = None
            if self.wanbd_logging:
                wandb.log({
                    'epoch': epoch + 1,
                    'total_loss': epoch_loss / len(train_loader),
                    'classification_loss': clf_epoch_loss / len(train_loader),
                    'reconstruction_loss': rec_epoch_loss / len(train_loader),
                    'val_total_loss': val_loss['total_loss'],
                    'val_classification_loss': val_loss['classification_loss'],
                    'val_reconstruction_loss': val_loss['reconstruction_loss'],
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'sp_index': sp,
                    'roc_auc': roc_auc,
                    'learning_rate': lr
                })

            if self.verbose:
                print(f"Epoch {epoch+1}/{self.num_epochs}, Total Loss: {epoch_loss/len(train_loader):.4f}, "
                      f"Classification Loss: {clf_epoch_loss/len(train_loader):.4f}, "
                      f"Reconstruction Loss: {rec_epoch_loss/len(train_loader):.4f}, "
                      f"Total Val Loss: {val_loss['total_loss']:.4f} "
                      f"\nClassification Val Loss: {val_loss['classification_loss']:.4f} "
                      f"Reconstruction Val Loss: {val_loss['reconstruction_loss']:.4f}, Accuracy: {accuracy:.4f}, "
                      f"Precision: {precision:.4f}, Recall: {recall}, "
                      f"F1 Score: {f1:.4f} "
                      f"SP Index: {sp:.4f}, ROC AUC: {roc_auc:.4f} "
                      f"Learning Rate = {lr}"
                )

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self.model
                

              
    def evaluate(self, test_loader):
        self.model.eval()
        all_preds = []
        all_targets = []
        val_total_loss = 0.0
        val_classification_loss = 0.0
        val_reconstruction_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_target in test_loader:
                class_output, reconstructed = self.model(batch_data)

                classification_loss = self.criterion(class_output, batch_target)
                reconstruction_loss = self.reconstruction_criterion(reconstructed, batch_data)

                total_loss = self.alpha * 10 * reconstruction_loss + (1 - self.alpha) * classification_loss
                val_total_loss += total_loss.item()
                val_classification_loss += classification_loss.item()
                val_reconstruction_loss += reconstruction_loss.item()
                
                _, preds = torch.max(class_output, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_target.cpu().numpy())

        val_total_loss /= len(test_loader)
        val_classification_loss /= len(test_loader)
        val_reconstruction_loss /= len(test_loader)
        
        y_pred = np.array(all_preds)
        y_target = np.array(all_targets)
        
        accuracy = np.mean(recall_score(y_target, y_pred, average=None))
        precision = precision_score(y_target, y_pred, average='weighted')
        recall = recall_score(y_target, y_pred, average=None)
        f1 = f1_score(y_target, y_pred, average='weighted')
        sp = sp_index(recall)
        roc_auc = roc_auc_score(
            label_binarize(y_target, classes=[0, 1, 2, 3]),
            label_binarize(y_pred, classes=[0, 1, 2, 3]),
            average='weighted',
            multi_class='ovr'
        )

        val_loss = {
            'total_loss': val_total_loss,
            'classification_loss': val_classification_loss,
            'reconstruction_loss': val_reconstruction_loss
        }
        return val_loss, accuracy, precision, recall, f1, sp, roc_auc
    
