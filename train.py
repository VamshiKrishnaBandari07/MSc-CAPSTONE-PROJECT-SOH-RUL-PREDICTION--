import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import BatteryHealthPredictor
from preprocess import BatteryDatasetLoader, RANDOM_SEED


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class BatteryDataset(Dataset):
    """
    Standard PyTorch Dataset for loading battery cycle features and target SOH and RUL labels.
    Normalizes SOH and RUL internally for stable model gradients.
    """
    def __init__(self, features, soh, rul, max_rul=150.0):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.soh = torch.tensor(soh, dtype=torch.float32).unsqueeze(1) # Shape: [N, 1]
        self.rul = torch.tensor(rul, dtype=torch.float32).unsqueeze(1) / max_rul # Shape: [N, 1], normalized
        self.max_rul = max_rul

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.soh[idx], self.rul[idx]


class JointPhysicsInformedLoss(nn.Module):
    """
    --- THESIS RESEARCH GAP ADDITION ---
    A custom loss function combining Multi-Task Learning (SOH & RUL estimation)
    with a Physics-Guided Monotonicity constraint for capacity fading.
    
    Loss = MSE_SOH + alpha * MSE_RUL + lambda_mono * Penalty_Mono
    """
    def __init__(self, rul_weight=0.5, monotonicity_weight=0.25):
        super(JointPhysicsInformedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.alpha = rul_weight
        self.lambda_mono = monotonicity_weight

    def forward(self, pred_soh, pred_rul, target_soh, target_rul):
        # 1. SOH Loss
        loss_soh = self.mse_loss(pred_soh, target_soh)
        
        # 2. RUL Loss
        loss_rul = self.mse_loss(pred_rul, target_rul)
        
        # 3. Physics Monotonicity Penalty
        # Since battery capacity decreases over time, consecutive predictions should not increase
        if pred_soh.size(0) > 1:
            diff = pred_soh[1:] - pred_soh[:-1] # SOH(t) - SOH(t-1)
            # Penalty for capacity recovery exceeding physical boundaries (i.e. positive diff)
            mono_penalty = torch.mean(torch.relu(diff))
        else:
            mono_penalty = torch.tensor(0.0, device=pred_soh.device)
            
        # Total Weighted Multi-Task Loss
        total_loss = loss_soh + self.alpha * loss_rul + self.lambda_mono * mono_penalty
        return total_loss, loss_soh, loss_rul, mono_penalty


def train_and_evaluate(dataset_name="NASA", num_epochs=10, batch_size=8, lr=1e-3):
    """
    Runs a complete training and validation cycle on a specified dataset.
    """
    print(f"\n==========================================")
    print(f"Starting Training on {dataset_name} Dataset...")
    print(f"==========================================")
    
    # 1. Load data
    raw_features, raw_soh, raw_rul = BatteryDatasetLoader.load_dataset(dataset_name, num_cycles=150, seq_len=100)
    
    # Train-test split (80% train, 20% validation)
    split_idx = int(len(raw_features) * 0.8)
    
    train_dataset = BatteryDataset(raw_features[:split_idx], raw_soh[:split_idx], raw_rul[:split_idx])
    val_dataset = BatteryDataset(raw_features[split_idx:], raw_soh[split_idx:], raw_rul[split_idx:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) # Order preserved for physics sequential checks
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Initialize model and optimization pipeline
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BatteryHealthPredictor(input_features=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Multi-task loss (balance RUL scale & monotonicity constraint)
    criterion = JointPhysicsInformedLoss(rul_weight=0.5, monotonicity_weight=0.25)
    
    # 3. Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_soh_loss = 0.0
        epoch_rul_loss = 0.0
        epoch_mono_loss = 0.0
        
        for features, targets_soh, targets_rul in train_loader:
            features = features.to(device)
            targets_soh = targets_soh.to(device)
            targets_rul = targets_rul.to(device)
            
            optimizer.zero_grad()
            pred_soh, pred_rul, _ = model(features)
            
            loss, l_soh, l_rul, l_mono = criterion(pred_soh, pred_rul, targets_soh, targets_rul)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * features.size(0)
            epoch_soh_loss += l_soh.item() * features.size(0)
            epoch_rul_loss += l_rul.item() * features.size(0)
            epoch_mono_loss += l_mono.item() * features.size(0)
            
        # 4. Validation step
        model.eval()
        val_soh_se = 0.0
        val_rul_se = 0.0
        
        with torch.no_grad():
            for features, targets_soh, targets_rul in val_loader:
                features = features.to(device)
                targets_soh = targets_soh.to(device)
                targets_rul = targets_rul.to(device)
                
                pred_soh, pred_rul, _ = model(features)
                
                # SOH squared error
                val_soh_se += nn.MSELoss(reduction='sum')(pred_soh, targets_soh).item()
                # RUL squared error (scale back to actual cycles)
                actual_pred_rul = pred_rul * train_dataset.max_rul
                actual_target_rul = targets_rul * train_dataset.max_rul
                val_rul_se += nn.MSELoss(reduction='sum')(actual_pred_rul, actual_target_rul).item()
                
        n_train = len(train_dataset)
        n_val = len(val_dataset)
        
        avg_loss = epoch_loss / n_train
        avg_soh_l = epoch_soh_loss / n_train
        avg_rul_l = epoch_rul_loss / n_train
        avg_mono_l = epoch_mono_loss / n_train
        
        val_soh_rmse = np.sqrt(val_soh_se / n_val)
        val_rul_rmse = np.sqrt(val_rul_se / n_val)
        
        print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | "
              f"Train Loss: {avg_loss:.5f} (SOH: {avg_soh_l:.4f}, RUL: {avg_rul_l:.4f}, Mono: {avg_mono_l:.4f}) | "
              f"Val SOH RMSE: {val_soh_rmse:.4f} | Val RUL RMSE: {val_rul_rmse:.2f} cycles")
              
    print(f"Training on {dataset_name} completed!")
    return val_soh_rmse, val_rul_rmse, model

def run_cross_validation():
    """
    Demonstrates training and cross-validation on NASA, Oxford, and CALCE datasets.
    """
    results = {}
    datasets = ["NASA", "Oxford", "CALCE"]
    
    for ds in datasets:
        soh_rmse, rul_rmse, _ = train_and_evaluate(dataset_name=ds, num_epochs=5, batch_size=8)
        results[ds] = {"SOH_RMSE": soh_rmse, "RUL_RMSE": rul_rmse}
        
    print("\n==========================================")
    print("Cross-Validation Benchmarks Summary:")
    print("==========================================")
    for ds, metrics in results.items():
        print(f"Dataset: {ds:<8} | SOH RMSE: {metrics['SOH_RMSE']:.4f} | RUL RMSE: {metrics['RUL_RMSE']:.2f} cycles")
    print("==========================================")

if __name__ == '__main__':
    set_seed()
    run_cross_validation()
