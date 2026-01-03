from trainer import Trainer
from utils.dataset import InformerDataset
from model import Informer
from utils.utils import initialize_weights

import pickle
import torch
from torch.utils.data import Dataset, DataLoader


folder = "/kaggle/working/Market-Data"
dataset = InformerDataset(
    data_dir =folder,
    seq_len  = 42,
    label_len= 30,
    pred_len =1,
    stride   = 25,
    max_features = 103,
    target_col="% Change (Pred)"
)

with open('/kaggle/working/logs/batch-size-32/scalers.pkl', 'wb') as f:
    pickle.dump(dataset.scalers, f)


batch_size = 32
train_size = int(0.80 * len(dataset))
test_size  = len(dataset) - train_size

train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
test_dataset  = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,num_workers = 4 ,pin_memory = True, persistent_workers = True, prefetch_factor = 4, drop_last = True)
valid_loader = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,num_workers = 4 ,pin_memory = True, persistent_workers = True, prefetch_factor = 4, drop_last = True) # Don't shuffle test data
print(f"Train loader samples: {len(train_loader)} | Test loader samples: {len(valid_loader)}")


model = Informer(
    enc_in=103,
    dec_in=103,
    c_out=1,

    factor=3,          
    d_model=64,        # VERY IMPORTANT
    n_heads=4,         # 64 / 4 = 16 (perfect)
    e_layers=2,
    d_layers=1,
    d_ff=4*64,          # 4 * 64
    dropout=0.30,      # higher dropout for small data

    attn='prob',
    distil=True
)

model.apply(initialize_weights)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device, non_blocking=True)

path    = '/kaggle/working/logs/batch-size-32'
trainer = Trainer(
    model,
    train_loader,
    valid_loader,
    device,
    batch_size=batch_size,
    num_epochs=16,
    lr=3e-4,
    min_lr=1e-5,
    weight_decay=1e-4
)
trainer.training(path)



