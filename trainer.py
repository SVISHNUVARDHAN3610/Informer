import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.utils import cosine_warmup_scheduler
from utils.utils import compute_regression_metrics
from utils.utils import ploting

class Trainer:
    def __init__(self, model, train_loader, valid_loader, device = 'cuda',batch_size = 32, num_epochs = 12, lr = 3e-4, min_lr = 3e-6, weight_decay = 0.01 ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device       = device
        self.num_epochs   = num_epochs
        self.lr           = lr
        self.min_lr       = min_lr
        self.weight_decay = weight_decay
        self.batch_size   = batch_size
        self.max_steps    = self.num_epochs * len(self.train_loader)
        self.warm_up      = int(0.05*self.max_steps)
        self.optimizer    = optim.AdamW(
                                self.model.parameters(),
                                lr = self.lr,
                                weight_decay = self.weight_decay
                            )
        self.scheduler = cosine_warmup_scheduler(
                            self.optimizer,
                            warmup_steps=self.warm_up,      
                            total_steps=self.max_steps,    
                            min_lr=self.min_lr            
                        )
        
        self.avg_loss   = []
        self.epoch_loss = []
        self.lr_cache   = []
        self.mse_list   = []
        self.rmse_list  = []
        self.mae_list   = []
        self.r2_scorelist = []
        self.valid_loss   = []
        self.grad_list    = []

    def loss_fn(self,y_pred,y_true):
        pred_last = y_pred[:, -1, :]     
        true_last = y_true[:, 0, :]      
        return F.smooth_l1_loss(pred_last, true_last, beta=0.1)

    def saving(self, path):
        torch.save(self.model.state_dict(), f'{path}/model-weights/model.pt')

    def evaulating(self, model, device):
        model.eval()
    
        preds, targets = [], []
        loss_list = []
    
        with torch.no_grad():
            for batch in self.valid_loader:
                src = batch["x_enc"].to(device,non_blocking=True)
                trg = batch["x_dec"].to(device,non_blocking=True)
                y   = batch["y"].to(device,non_blocking=True)
    
                pred = model(src, trg)
                pred_last = pred[:, -1:, :]
    
                loss = self.loss_fn(pred, y)
                loss_list.append(loss.detach().cpu())
    
                preds.append(pred_last.detach().cpu())
                targets.append(y.detach().cpu())
    
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
    
        metrics = compute_regression_metrics(preds, targets)
    
        val_loss = sum(loss_list) / len(loss_list)
        smoothed_r2 = np.mean(self.r2_scorelist[-3:])
        print(
            f"Validation: Loss: {val_loss:.4f} || "
            f"MSE: {metrics['mse']:.5f} || "
            f"MAE: {metrics['mae']:.4f}  || "
            f"RMSE: {metrics['rmse']:.4f} || "
            f"R2: {metrics['r2_score']:.4f} ||" 
            f"smoothedR2: {smoothed_r2:.4f}"
        )
    
        self.valid_loss.append(val_loss)
        self.mse_list.append(metrics["mse"])
        self.mae_list.append(metrics["mae"])
        self.rmse_list.append(metrics["rmse"])
        self.r2_scorelist.append(metrics["r2_score"])
    
        model.train()
    def ploting(self,epoch,path):
        ploting(epoch,path,
                self.lr_cache, 
                self.epoch_loss,
                self.avg_loss,
                self.valid_loss,
                self.grad_list,
                self.mae_list,
                self.mse_list,
                self.rmse_list,
                self.r2_scorelist,
                self.batch_size)
    def training(self,path):
        print(f'Training started with data loder size of {len(self.train_loader)}.......')
        scaler = torch.amp.GradScaler(enabled=True)
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for step, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                x_enc = batch['x_enc'].to(self.device,non_blocking=True)
                x_dec = batch['x_dec'].to(self.device,non_blocking=True)
                y_true = batch['y'].to(self.device,non_blocking=True)

                y_pred = self.model(x_enc, x_dec)
                loss = self.loss_fn(y_pred, y_true)
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()
                torch.cuda.synchronize()
                self.scheduler.step()
                temp = loss.detach().item()
                epoch_loss += temp
                current_lr = self.optimizer.param_groups[0]['lr']
                self.lr_cache.append(current_lr)
                self.epoch_loss.append(temp)
            print(f"Training  : Epoch: {epoch+1}/{self.num_epochs}  || Loss: {epoch_loss/len(self.train_loader) :.3f}  || Grad: {grad_norm :.4f} || LR: {current_lr:.6f} ||")
            self.grad_list.append(grad_norm.cpu())
            self.avg_loss.append(epoch_loss/len(self.train_loader))
            self.saving(path)
            self.evaulating(self.model,self.device)
            self.ploting(epoch,path)
            
        print("Training completed successfully!")