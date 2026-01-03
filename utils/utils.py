import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def initialize_weights(module):

    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)

def compute_regression_metrics(preds, targets):
    """
    preds: shape (B, L, 1)
    targets: shape (B, L, 1)
    """

    preds = np.array(preds).squeeze()     # (1,16,1) → (16,)
    targets = np.array(targets).squeeze() # (1,16,1) → (16,)

    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    rmse = mse ** 0.5
    r2 = r2_score(targets, preds)

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2_score": float(r2)
    }

def cosine_warmup_scheduler(optimizer, warmup_steps, total_steps, min_lr=7e-6):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return max(
            min_lr,
            0.5 * (1 + math.cos(math.pi * progress))
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def ploting(epoch,path,lr_cache, epoch_loss,avg_loss,valid_loss,grad_list,mae_list,mse_list,rmse_list,r2_scorelist,batch_size):
    def lr_ploting(epoch,path):
      plt.plot(lr_cache, label='Learning Rate')
      plt.xlabel('steps')
      plt.ylabel('Learning Rate')
      plt.title('Learning Rate Schedule')
      plt.savefig(f'{path}/ploting/lr_plot_bs_{batch_size}.png')
      plt.close()
        
    def epoch_loss_ploting(epoch,path):
      plt.plot(epoch_loss, label='Loss')
      plt.xlabel('Steps')
      plt.ylabel('Loss')
      plt.title(f'Loss vs steps on epoch {epoch} with batch_size {batch_size}')
      plt.savefig(f'{path}/ploting/temp/epoch_loss_plot_bs_{batch_size}_epoch_{epoch}.png')
      plt.close()
    def avg_loss_ploting(epoch,path):
      plt.plot(avg_loss, label='Loss')
      plt.xlabel('epochs')
      plt.ylabel('Loss')
      plt.title(f'Loss vs epochs with batch_size {batch_size}')
      plt.savefig(f'{path}/ploting/avg_loss_plot_bs_{batch_size}.png')
      plt.close()
    def grad_ploting(epoch,path):
      plt.plot(grad_list, label='Loss')
      plt.xlabel('epochs')
      plt.ylabel('Gradient')
      plt.title(f'gradient vs epochs with batch_size {batch_size}')
      plt.savefig(f'{path}/ploting/temp/gradient_plot_bs_{batch_size}.png')
      plt.close()
        
    def valid_ploting(epoch,path):
        epochs = [i for i in range(epoch+1)]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Validation Metrics History (Epochs 0-15)", fontsize=16)
        
        # ---- MAE ----
        axes[0, 0].plot(epochs, mae_list, marker='o')
        axes[0, 0].set_title("MAE (Mean Absolute Error)")
        axes[0, 0].set_xlabel("Epochs")
        axes[0, 0].set_ylabel("Error")
        axes[0, 0].grid(True)
        axes[0, 0].legend(["MAE (Mean Absolute Error)", "Fine-Tuning Start"])
        
        # ---- MSE ----
        axes[0, 1].plot(epochs, mse_list, marker='o')
        axes[0, 1].set_title("MSE (Mean Squared Error)")
        axes[0, 1].set_xlabel("Epochs")
        axes[0, 1].set_ylabel("Error Squared")
        axes[0, 1].grid(True)
        axes[0, 1].legend(["MSE (Mean Squared Error)", "Fine-Tuning Start"])
        
        # ---- RMSE ----
        axes[1, 0].plot(epochs, rmse_list, marker='o')
        axes[1, 0].set_title("RMSE (Root MSE)")
        axes[1, 0].set_xlabel("Epochs")
        axes[1, 0].set_ylabel("Error")
        axes[1, 0].grid(True)
        axes[1, 0].legend(["RMSE (Root MSE)", "Fine-Tuning Start"])
        
        # ---- R2 ----
        axes[1, 1].plot(epochs, r2_scorelist, marker='o')
        axes[1, 1].set_title("R2 Score")
        axes[1, 1].set_xlabel("Epochs")
        axes[1, 1].set_ylabel("Score")
        axes[1, 1].grid(True)
        axes[1, 1].legend(["R2 Score", "Fine-Tuning Start"])
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'{path}/ploting/valid_plot.png')
        plt.close()
    def train_valid(epoch,path):
        epochs = np.arange(len(avg_loss))
        if len(avg_loss) == 0 or len(valid_loss) == 0:
            return
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(epochs, avg_loss,
                 marker='o', linewidth=2,
                 label='Train Loss')
        
        plt.plot(epochs, valid_loss,
                 marker='s', linewidth=2,
                 linestyle='--',
                 label='Validation Loss')
        
        # Styling
        plt.title("Training vs Validation Loss", fontsize=12)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(f'{path}/ploting/train_valid_plot.png')
        plt.close()
        
    lr_ploting(epoch,path)
    epoch_loss_ploting(epoch,path)
    avg_loss_ploting(epoch,path)
    valid_ploting(epoch,path)
    train_valid(epoch,path)
    grad_ploting(epoch,path)