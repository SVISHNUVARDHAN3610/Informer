import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import glob
import pandas as pd
import numpy as np
import os, sys

from torch.utils.data import Dataset, DataLoader

class InformerDataset(Dataset):
    def __init__(self,
                 data_dir,
                 seq_len=5,
                 label_len=4,
                 pred_len=1,
                 target_col="% Change",
                 max_features=103,
                 stride=1):  # <--- CHANGE 1: Accept stride parameter from user
        
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.target_col = target_col
        self.max_features = max_features
        self.stride = stride  # <--- CHANGE 2: Store stride
        
        self.all_files = glob.glob(os.path.join(data_dir, "*.csv"))
        self.index_map = []
        self.data_cache = []
        self.scalers = {} 

        print(f"Building Informer Dataset from {len(self.all_files)} files with Stride={self.stride}...")

        for file_idx, file_path in enumerate(self.all_files):
            try:
                df = pd.read_csv(file_path)
                
                # Minimum length check
                min_len = self.seq_len + self.pred_len
                if df.empty or len(df) < min_len:
                    continue

                # Drop Date
                if 'Date' in df.columns:
                    df = df.drop(columns=['Date'])
                
                # Identify Target Column
                active_target = self.target_col
                if active_target not in df.columns:
                    df.columns = df.columns.str.strip() 
                    if active_target not in df.columns:
                        if '% Change' in df.columns: active_target = '% Change'
                        else: continue

                # Normalization Logic
                numeric_cols = df.select_dtypes(include=['float32', 'float64', 'int64', 'int32']).columns.tolist()
                subset = df[numeric_cols]
                mean_vals = subset.mean()
                std_vals = subset.std().replace(0, 1.0)
                
                df[numeric_cols] = (subset - mean_vals) / std_vals
                
                target_mean = mean_vals[active_target]
                target_std = std_vals[active_target]
                
                features_data = df[numeric_cols].fillna(0).values
                target_data = df[active_target].fillna(0).values

                feat_tensor = torch.tensor(features_data, dtype=torch.float32)
                target_tensor = torch.tensor(target_data, dtype=torch.float32).unsqueeze(1)

                # Feature Padding/Cutting
                curr_cols = feat_tensor.shape[1]
                if curr_cols > self.max_features:
                    feat_tensor = feat_tensor[:, :self.max_features]
                elif curr_cols < self.max_features:
                    pad_amt = self.max_features - curr_cols
                    feat_tensor = F.pad(feat_tensor, (0, pad_amt), "constant", 0)

                self.data_cache.append((feat_tensor, target_tensor))

                num_rows = len(df)
                cache_idx = len(self.data_cache) - 1 
                
                self.scalers[cache_idx] = {'mean': target_mean, 'std': target_std}

                # ---------------------------------------------------------
                # CHANGE 3: Apply Stride in the Range Loop
                # ---------------------------------------------------------
                # Calculate the upper limit for the loop
                limit = num_rows - self.seq_len - self.pred_len + 1
                
                # Use the 3-argument range: range(start, stop, step)
                # This automatically skips indices based on self.stride
                for start_idx in range(0, limit, self.stride):
                    self.index_map.append((cache_idx, start_idx))

            except Exception as e:
                print(f"Skipping {os.path.basename(file_path)}: {e}")

        print(f"Dataset Ready. Total Samples: {len(self.index_map)}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # No changes needed here! 
        # Since index_map only contains the "strided" indices, 
        # this method just grabs them directly.
        file_idx, s_begin = self.index_map[idx]
        data_feat, data_target = self.data_cache[file_idx]
        stats = self.scalers[file_idx]

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        enc_x = data_feat[s_begin:s_end]
        x_dec_label = data_feat[r_begin:s_end]
        x_dec_zeros = torch.zeros(self.pred_len, self.max_features)
        dec_x = torch.cat([x_dec_label, x_dec_zeros], dim=0)
        
        target = data_target[s_end : s_end + self.pred_len]

        return {
            'x_enc': enc_x,
            'x_dec': dec_x,
            'y': target,
            'mean': torch.tensor(stats['mean'], dtype=torch.float32),
            'std': torch.tensor(stats['std'], dtype=torch.float32)
        }