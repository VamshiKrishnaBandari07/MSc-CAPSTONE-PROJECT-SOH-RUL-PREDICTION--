import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from model_paper import BatterySOHPredictorPaper
from preprocess_paper import generate_paper_synthetic_data

class BatteryDatasetPaper(Dataset):
    def __init__(self, features, soh):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.soh = torch.tensor(soh, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.soh[idx]


def train_exact_paper_model(epochs=5, batch_size=8):
    """
    EXACT PAPER TRAINING PIPELINE:
    - Target: State of Health (SOH) only.
    - Loss: Pure Mean Squared Error (MSE).
    - Optimizer: Standard Adam Optimizer.
    """
    print("\n==========================================")
    print("Starting Exact Paper Model Training Run (Pure SOH, MSE Loss)...")
    print("==========================================")
    
    # 1. Generate data
    raw_features, raw_soh = generate_paper_synthetic_data(num_cycles=150, seq_len=100)
    
    split_idx = int(len(raw_features) * 0.8)
    
    train_dataset = BatteryDatasetPaper(raw_features[:split_idx], raw_soh[:split_idx])
    val_dataset = BatteryDatasetPaper(raw_features[split_idx:], raw_soh[split_idx:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Init model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BatterySOHPredictorPaper(input_features=3).to(device)
    
    # Pure data-driven SOH MSE Loss as in the paper
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # 3. Train Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            pred, _ = model(features)
            
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            
        # Validation
        model.eval()
        val_se = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                pred, _ = model(features)
                val_se += nn.MSELoss(reduction='sum')(pred, targets).item()
                
        n_train = len(train_dataset)
        n_val = len(val_dataset)
        
        avg_train_loss = train_loss / n_train
        val_rmse = np.sqrt(val_se / n_val)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train MSE Loss: {avg_train_loss:.5f} | Val SOH RMSE: {val_rmse:.4f}")
        
    print("Exact Paper Model Training Completed!")
    return val_rmse, model

if __name__ == '__main__':
    train_exact_paper_model(epochs=5, batch_size=8)
