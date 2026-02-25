# for submission to ICLR 2026


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from decoder.model import NNDecoder


dpi = 300


# set device
def set_device(verbose=True):
    """
    Set the device. CUDA if available, CPU otherwise

    Args:
    None

    Returns:
    Nothing
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose == True:
        if device != "cuda":
            print("WARNING: For this notebook to perform best, "
                "if possible, in the menu under `Runtime` -> "
                "`Change runtime type.`  select `GPU` ")
        else:
            print("GPU is enabled in this session.")

    return device


# take log prob to prob space and normalize
def normalize_exp(arr):
    arr_exp = np.exp(arr)
    arr_exp_normalized = arr_exp/ np.sum(arr_exp)
    return arr_exp_normalized


# -- training --
class Trainer():
    def __init__(self, args):
        self.args = args

        # model initialization
        self.decoder = NNDecoder(
            input_dim=self.args.input_dim,
            layers=self.args.nn_layers,
            dropout_rates=self.args.nn_dropout_rates,
            output_dim=self.args.output_dim
        ).to(self.args.device)

        # initialize optimizer
        self.criterion = nn.CrossEntropyLoss().to(self.args.device)
        self.optimizer = torch.optim.Adam(
            self.decoder.parameters(), 
            lr=self.args.learning_rate
        )

        # prepare prior
        if self.args.decoder_type == 'flex':
            self.fixed_log_prior_reference = torch.zeros(
                self.args.output_dim, requires_grad=False
            ).to(self.args.device)
            self.flex_log_prior_diff = torch.zeros(
                self.args.output_dim, requires_grad=True
            ).to(self.args.device)
            print(f'initial flexible log prior diff: {self.flex_log_prior_diff}')
            self.optimizer.add_param_group({"params": self.flex_log_prior_diff})
        
        # Early stopping state
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        self.best_flex_prior_state = None

    def _save_best_model(self):
        """Save the current best model state"""
        self.best_model_state = {
            'decoder': self.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        if self.args.decoder_type == 'flex':
            self.best_flex_prior_state = self.flex_log_prior_diff.clone().detach()

    def _restore_best_model(self):
        """Restore the best model state"""
        if self.best_model_state is not None:
            self.decoder.load_state_dict(self.best_model_state['decoder'])
            self.optimizer.load_state_dict(self.best_model_state['optimizer'])
            if self.args.decoder_type == 'flex' and self.best_flex_prior_state is not None:
                self.flex_log_prior_diff.data.copy_(self.best_flex_prior_state)

    def _check_early_stopping(self, current_loss, epoch):
        """
        Check if training should stop early based on loss
        
        Args:
            current_loss (float): Current epoch's loss
            epoch (int): Current epoch number
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.args.early_stopping_patience is None:
            return False
            
        # Check if current loss is better than best loss
        if current_loss < self.best_loss - self.args.early_stopping_min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
            self._save_best_model()
            if epoch % 100 == 0:  # Only print on verbose epochs
                print(f'  -> New best loss: {current_loss:.6f}')
        else:
            self.patience_counter += 1
            
        # Check if we should stop
        if self.patience_counter >= self.args.early_stopping_patience:
            print(f'Early stopping triggered at epoch {epoch} (patience: {self.patience_counter})')
            if self.args.early_stopping_restore_best:
                print('Restoring best model weights')
                self._restore_best_model()
            return True
            
        return False

    def _validate(self, validation_dataset, validation_sampler=None):
        """
        Perform validation and return average loss
        
        Args:
            validation_dataset: Validation dataset
            validation_sampler: Optional validation sampler
            
        Returns:
            float: Average validation loss
        """
        self.decoder.eval()
        total_loss = 0
        num_batches = 0
        
        # Create validation data loader
        if validation_sampler is not None:
            val_loader = DataLoader(
                validation_dataset, 
                sampler=validation_sampler, 
                batch_size=self.args.batch_size
            )
        else:
            val_loader = DataLoader(
                validation_dataset, 
                batch_size=self.args.batch_size, 
                shuffle=False
            )
        
        with torch.no_grad():
            for x_, t_ in val_loader:
                t = t_.to(self.args.device)
                x_ = x_.type(torch.FloatTensor).to(self.args.device)
                x = x_[:, :self.args.input_dim].to(self.args.device)
                
                # Same logic as training for computing log_priors
                if self.args.decoder_type == 'lh':
                    priors = x_[:, self.args.input_dim+1:].to(self.args.device)
                    eps = 1e-8
                    priors = torch.clamp(priors, min=eps)
                    priors = priors / priors.sum(dim=1, keepdim=True)
                    log_priors = torch.log(priors)
                elif self.args.decoder_type == 'flex':
                    task_ids = x_[:, self.args.input_dim].int()
                    fixed_index = (task_ids == 0).nonzero(as_tuple=False).squeeze()
                    flexible_index = (task_ids == 1).nonzero(as_tuple=False).squeeze()
                    log_priors = torch.empty(x.shape[0], self.args.output_dim)
                    if fixed_index.numel() > 0:
                        log_priors[fixed_index] = self.fixed_log_prior_reference
                    if flexible_index.numel() > 0:
                        log_priors[flexible_index] = self.flex_log_prior_diff
                    log_priors = log_priors.to(self.args.device)
                
                # Forward pass
                if self.args.decoder_type == 'post':
                    y = self.decoder(x)
                    log_post = y
                elif self.args.decoder_type in ['lh', 'flex']:
                    y = self.decoder(x)
                    log_post = y + log_priors
                
                # Shift to avoid numerical overflow
                val, _ = log_post.max(1, keepdim=True)
                log_post = log_post - val
                
                # Compute losses
                loss_cross_entropy_batch = self.criterion(log_post, t)
                
                # Smoothness loss
                conv_filter = torch.from_numpy(
                    np.array([-0.25, 0.5, -0.25])[None, None, :]).type(log_post.data.type())
                try:
                    loss_smoothness_batch = nn.functional.conv1d(log_post.unsqueeze(1), conv_filter).pow(2).mean()
                except:
                    loss_smoothness_batch = torch.tensor(0.0).to(self.args.device)
                
                loss_batch = loss_cross_entropy_batch + \
                    self.args.loss_smoothness_coeff * loss_smoothness_batch
                
                total_loss += loss_batch.item()
                num_batches += 1
        
        self.decoder.train()  # Switch back to training mode
        return total_loss / num_batches if num_batches > 0 else float('inf')

    def train(
        self, 
        dataset, 
        data_sampler=None,
        validation_dataset=None,
        validation_sampler=None
    ):
        if self.args.decoder_type not in ['lh', 'post', 'flex']:
            raise ValueError(f'unsupported decoder type: {self.args.decoder_type}')
        
        losses_cross_entropy = []
        losses_smoothness = []
        validation_losses = []
        self.decoder.train()

        for epoch in range(int(self.args.num_epochs)):
            loss_cross_entropy_epoch, loss_smoothness_epoch = 0, 0

            # load data
            if data_sampler is not None:
                train_loader = DataLoader(
                    dataset, 
                    sampler=data_sampler, 
                    batch_size=self.args.batch_size
                )
            else:
                train_loader = DataLoader(
                    dataset, 
                    batch_size=self.args.batch_size, 
                    shuffle=True
                )
            
            for x_, t_ in train_loader:
                t = t_.to(self.args.device)  # (batch_size)
                
                x_ = x_.type(torch.FloatTensor).to(self.args.device)  # (batch_size, n_units + 1 + n_bins)
                x = x_[:, :self.args.input_dim].to(self.args.device)  # (batch_size, n_units)
            
                if self.args.decoder_type == 'lh':
                    priors = x_[:, self.args.input_dim+1:].to(self.args.device)
                    # Add epsilon to prevent log(0) and clamp to ensure positive values
                    eps = 1e-8
                    priors = torch.clamp(priors, min=eps)

                    # Re-normalize after clamping to ensure they sum to 1
                    priors = priors / priors.sum(dim=1, keepdim=True)

                    log_priors = torch.log(priors)  # (batch_size, n_bins)

                elif self.args.decoder_type == 'flex':
                    task_ids = x_[:, self.args.input_dim].int()
                    fixed_index = (task_ids == 0).nonzero(as_tuple=False).squeeze()
                    flexible_index = (task_ids == 1).nonzero(as_tuple=False).squeeze()
                    log_priors = torch.empty(x.shape[0], self.args.output_dim)
                    if fixed_index.numel() > 0:
                        log_priors[fixed_index] = self.fixed_log_prior_reference
                    if flexible_index.numel() > 0:
                        log_priors[flexible_index] = self.flex_log_prior_diff
                    log_priors = log_priors.to(self.args.device)  # (batch_size, n_bins)

                # -- training --
                self.optimizer.zero_grad()
                if self.args.decoder_type == 'post':
                    # model output  -> log posterior
                    y = self.decoder(x)  
                    log_post = y
                elif self.args.decoder_type in ['lh', 'flex']:
                    # model output  -> log likelihood
                    y = self.decoder(x) 
                    log_post = y + log_priors  # adding log prior -> log posterior
                
                # shift to avoid numerical overflow
                val, _ = log_post.max(1, keepdim=True)
                log_post = log_post - val
                
                # encouraging smoothness
                conv_filter = torch.from_numpy(
                    np.array([-0.25, 0.5, -0.25])[None, None, :]).type(log_post.data.type())
                try:
                    loss_smoothness_batch = nn.functional.conv1d(log_post.unsqueeze(1), conv_filter).pow(2).mean()
                except:
                    # if smoothness computation overflows, then don't bother with it
                    loss_smoothness_batch = 0

                loss_cross_entropy_batch = self.criterion(log_post, t)
                loss_batch = loss_cross_entropy_batch + \
                    self.args.loss_smoothness_coeff * loss_smoothness_batch
                
                loss_batch.backward()
                self.optimizer.step()
                
                loss_cross_entropy_epoch += loss_cross_entropy_batch.data.cpu().numpy()
                if hasattr(loss_smoothness_batch, 'data'):
                    loss_smoothness_epoch += loss_smoothness_batch.data.cpu().numpy()
                else:
                    loss_smoothness_epoch += loss_smoothness_batch

            # Calculate epoch losses
            train_loss_ce = loss_cross_entropy_epoch / float(len(dataset))
            train_loss_smooth = loss_smoothness_epoch / float(len(dataset))
            train_loss_total = train_loss_ce + self.args.loss_smoothness_coeff * train_loss_smooth
            
            losses_cross_entropy.append(train_loss_ce)
            losses_smoothness.append(train_loss_smooth)

            # Validation phase (if validation data provided)
            val_loss_total = None
            if validation_dataset is not None:
                val_loss_total = self._validate(validation_dataset, validation_sampler)
                validation_losses.append(val_loss_total)
                # Use validation loss for early stopping if available
                early_stop_loss = val_loss_total
            else:
                # Use training loss for early stopping if no validation data
                early_stop_loss = train_loss_total

            # Check early stopping
            if self.args.early_stopping:
                if self._check_early_stopping(early_stop_loss, epoch):
                    print(f'Training stopped early at epoch {epoch}')
                    # Trim losses arrays to actual length
                    losses_cross_entropy = losses_cross_entropy[:epoch+1]
                    losses_smoothness = losses_smoothness[:epoch+1]
                    if validation_losses:
                        validation_losses = validation_losses[:epoch+1]
                    break

        # after training
        losses_cross_entropy = np.array(losses_cross_entropy)
        losses_smoothness = np.array(losses_smoothness)
        losses_total = losses_cross_entropy + self.args.loss_smoothness_coeff * losses_smoothness
        validation_losses = np.array(validation_losses) if validation_losses else np.array([None])

        if self.args.decoder_type in ['lh', 'post']:
            self.flex_log_prior_diff = np.array([None])
        else:
            print(f'fixed prior: {self.fixed_log_prior_reference}')
            print(f'trained flexible prior: {self.flex_log_prior_diff}')
            self.flex_log_prior_diff = self.flex_log_prior_diff.data.cpu().numpy()

        # return dict
        return {
            'losses_total': losses_total,
            'losses_cross_entropy': losses_cross_entropy,
            'losses_smoothness': losses_smoothness,
            'validation_losses': validation_losses,
            'flex_log_prior_diff': self.flex_log_prior_diff
        }
        

    def plot_training_losses(
        self,
        losses_total=None,
        losses_cross_entropy=None, 
        losses_smoothness=None,
        validation_losses=None,
        figsize=(6, 4),
        show_plot=False        
    ):
        """
        Plot all training losses on the same axis for comparison.
        
        Args:
            losses_total (np.array, optional): Total training losses per epoch
            losses_cross_entropy (np.array, optional): Cross-entropy losses per epoch
            losses_smoothness (np.array, optional): Smoothness losses per epoch
            save_path (str, optional): Path to save the plot. If None, plot is not saved
            show_plot (bool): Whether to display the plot
            figsize (tuple): Figure size as (width, height)
            
        Returns:
            matplotlib.figure.Figure: The created figure object
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.set_title('Training Losses')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        
        # Plot each loss type if provided
        if losses_total is not None:
            epochs = range(len(losses_total))
            ax.plot(epochs, losses_total, 'r-', linewidth=1, label='Total Loss')
            
        if losses_cross_entropy is not None:
            epochs = range(len(losses_cross_entropy))
            ax.plot(epochs, losses_cross_entropy, 'b-', linewidth=1, label='Cross-Entropy Loss')
            
        if losses_smoothness is not None:
            epochs = range(len(losses_smoothness))
            ax.plot(epochs, losses_smoothness, 'g-', linewidth=1, label='Smoothness Loss')
        
        if validation_losses is not None:
            epochs = range(len(validation_losses))
            ax.plot(epochs, validation_losses, 'k--', linewidth=1, label='Validation Loss')

        ax.legend()
        ax.grid(True, alpha=0.3)        
        plt.tight_layout()
            
        # Show plot
        if show_plot:
            plt.show()
            
        return fig