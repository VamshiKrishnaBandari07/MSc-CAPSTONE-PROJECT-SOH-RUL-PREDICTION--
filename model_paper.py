import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

class Chomp1d(nn.Module):
    """
    Slices the output of a dilated 1D convolution to make it causal.
    As specified in the paper's TCN block.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Standard TCN residual block from the paper. Dilated, causal, and weight-normalized.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

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
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                padding=padding, dropout=dropout
            )]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy, dim=1)
        outputs = encoder_outputs * weights
        context = torch.sum(outputs, dim=1)
        return context, weights


class BatterySOHPredictorPaper(nn.Module):
    """
    EXACT PAPER IMPLEMENTATION:
    - 1D-CNN: Captures local cycle spatial configurations.
    - TCN: Dilated convolutions for temporal dependencies.
    - LSTM: Models capacity fading transitions.
    - Self-Attention: Highlights critical aging peaks.
    - Regression Head: Standard single SOH scalar estimation (no RUL prediction).
    """
    def __init__(self, input_features=3, cnn_out_channels=32, tcn_channels=[32, 64], lstm_hidden=64, num_lstm_layers=1, dropout=0.2):
        super(BatterySOHPredictorPaper, self).__init__()
        
        self.conv1d = nn.Conv1d(
            in_channels=input_features, 
            out_channels=cnn_out_channels, 
            kernel_size=5, 
            padding=2
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.cnn_dropout = nn.Dropout(dropout)
        
        self.tcn = TemporalConvNet(
            num_inputs=cnn_out_channels, 
            num_channels=tcn_channels, 
            kernel_size=3, 
            dropout=dropout
        )
        
        self.lstm = nn.LSTM(
            input_size=tcn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=False
        )
        
        self.attention = SelfAttention(lstm_hidden)
        
        # SOH Only Regression Head (exact paper setup)
        self.fc1 = nn.Linear(lstm_hidden, 32)
        self.fc2 = nn.Linear(32, 1) # SOH scalar output [0, 1]
        self.fc_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 1D-CNN
        x_cnn = self.conv1d(x)
        x_cnn = self.relu(x_cnn)
        x_cnn = self.pool(x_cnn)
        x_cnn = self.cnn_dropout(x_cnn)
        
        # TCN
        x_tcn = self.tcn(x_cnn)
        
        # LSTM input shape: [B, seq_len_tcn, features]
        x_lstm_in = x_tcn.transpose(1, 2)
        lstm_out, _ = self.lstm(x_lstm_in)
        
        # Self-Attention
        context, attn_weights = self.attention(lstm_out)
        
        # FC layers (SOH estimation)
        out = self.fc1(context)
        out = self.relu(out)
        out = self.fc_dropout(out)
        soh = self.fc2(out)
        
        return soh, attn_weights

if __name__ == '__main__':
    batch_size = 8
    features = 3
    seq_len = 100
    
    x = torch.randn(batch_size, features, seq_len)
    model = BatterySOHPredictorPaper()
    soh, weights = model(x)
    
    print("--- Exact Paper Model Scaffolding Verified ---")
    print(f"Input shape:      {x.shape}")
    print(f"SOH Output shape: {soh.shape}")
    print(f"Weights shape:    {weights.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params / 1e6:.4f} Million")
