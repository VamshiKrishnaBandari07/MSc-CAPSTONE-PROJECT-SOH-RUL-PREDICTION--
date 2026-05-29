import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

class Chomp1d(nn.Module):
    """
    Slices the output of a dilated 1D convolution to make it causal.
    Used in Temporal Convolutional Networks (TCN) to prevent looking into the future.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    A single residual block of a Temporal Convolutional Network (TCN).
    Consists of dilated causal convolutions, weight normalization, ReLU, and residual connection.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # First conv block
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second conv block
        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) module.
    Stacks multiple TemporalBlocks with exponentially growing dilations.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            # Padding is calculated to maintain the causal property
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                padding=padding, dropout=dropout
            )]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SelfAttention(nn.Module):
    """
    Self-Attention Mechanism to align and focus on the most critical
    electrochemical features and aging states.
    """
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )

    def forward(self, encoder_outputs):
        # encoder_outputs shape: [batch_size, seq_len, hidden_dim]
        energy = self.projection(encoder_outputs) # [batch_size, seq_len, 1]
        weights = F.softmax(energy, dim=1) # [batch_size, seq_len, 1]
        
        # Context vector
        outputs = encoder_outputs * weights # [batch_size, seq_len, hidden_dim]
        context = torch.sum(outputs, dim=1) # [batch_size, hidden_dim]
        
        return context, weights


class BatteryHealthPredictor(nn.Module):
    """
    State-of-the-Art Hybrid CNN-TCN-LSTM-Attention Predictor
    Matches the 2026 Nature Scientific Reports Paper:
    - 1D CNN: Extracts local spatial peaks and shapes from dQ/dV, dV/dQ, dI/dV profiles.
    - TCN: Captures medium-term causal degradation dynamics.
    - LSTM: Models long-term cycle-to-cycle capacity fading.
    - Attention: Identifies and weights crucial degradation phases.
    - Joint Predictor: SOH regression head & RUL regression head.
    """
    def __init__(self, input_features=3, cnn_out_channels=32, tcn_channels=[32, 64], lstm_hidden=64, num_lstm_layers=1, dropout=0.2):
        super(BatteryHealthPredictor, self).__init__()
        
        # 1. 1D CNN for local feature extraction (extracts shapes from raw cycles)
        self.conv1d = nn.Conv1d(
            in_channels=input_features, 
            out_channels=cnn_out_channels, 
            kernel_size=5, 
            padding=2
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) # Downsample dimension
        self.relu = nn.ReLU()
        self.cnn_dropout = nn.Dropout(dropout)
        
        # 2. TCN for causal medium-range sequential modeling
        self.tcn = TemporalConvNet(
            num_inputs=cnn_out_channels, 
            num_channels=tcn_channels, 
            kernel_size=3, 
            dropout=dropout
        )
        
        # 3. LSTM for long-horizon cycle-to-cycle dependencies
        self.lstm = nn.LSTM(
            input_size=tcn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # 4. Self-Attention mechanism
        self.attention = SelfAttention(lstm_hidden)
        
        # 5. Joint regression heads
        # SOH prediction: outputs a single scalar capacity ratio [0.0 - 1.0]
        self.soh_head = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        # RUL prediction: outputs a single scalar cycle count (e.g. 0 to 500)
        self.rul_head = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        Forward pass input shape x: [batch_size, input_features, seq_len]
        - input_features: (ICA dQ/dV, DVA dV/dQ, DCA dI/dV)
        - seq_len: Length of voltage/capacity steps in one charging cycle.
        """
        # --- 1. Spatial 1D Convolution ---
        # Input x:         [B, 3, L]
        # Out Conv1d:      [B, 32, L]
        # Out Pool:        [B, 32, L // 2]
        x_cnn = self.conv1d(x)
        x_cnn = self.relu(x_cnn)
        x_cnn = self.pool(x_cnn)
        x_cnn = self.cnn_dropout(x_cnn)
        
        # --- 2. Temporal Convolutional Network (TCN) ---
        # Out TCN:         [B, 64, L // 2] (based on tcn_channels[-1]=64)
        x_tcn = self.tcn(x_cnn)
        
        # Reshape for LSTM: [B, L // 2, 64]
        x_lstm_in = x_tcn.transpose(1, 2)
        
        # --- 3. Long Short-Term Memory (LSTM) ---
        # Out LSTM:        [B, L // 2, 64]
        lstm_out, _ = self.lstm(x_lstm_in)
        
        # --- 4. Attention Mechanism ---
        # context:         [B, 64]
        # attn_weights:    [B, L // 2, 1]
        context, attn_weights = self.attention(lstm_out)
        
        # --- 5. Joint Regressors ---
        # soh:             [B, 1]
        # rul:             [B, 1]
        soh = self.soh_head(context)
        rul = self.rul_head(context)
        
        return soh, rul, attn_weights

if __name__ == '__main__':
    # Dry run verification to check parameters and structure
    batch_size = 8
    features = 3     # (dQ/dV, dV/dQ, dI/dV)
    seq_len = 100    # 100 voltage measurement steps per cycle
    
    x = torch.randn(batch_size, features, seq_len)
    model = BatteryHealthPredictor()
    
    soh, rul, weights = model(x)
    
    print("--- Model Verification Run Successful ---")
    print(f"Input Shape:      {x.shape}")
    print(f"Output SOH Shape: {soh.shape}")
    print(f"Output RUL Shape: {rul.shape}")
    print(f"Attention Weights Shape: {weights.shape}")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params / 1e6:.3f} Million (Paper Target: ~0.35M)")
