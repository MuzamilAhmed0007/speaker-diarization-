# models/titanet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )

    def forward(self, x):
        return self.transformer(x)


class TitaNet(nn.Module):
    """
    TitaNet model for speaker embedding generation (E = f(x; θ)),
    where x is the mel-spectrogram input from sum of speakers + background noise:
    x = sum(s_i) + n
    """
    def __init__(self, input_dim=80, embedding_dim=192):
        super(TitaNet, self).__init__()

        # Input CNN layers (learn x_e = encoded features from x)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            SEBlock(128),

            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # Bidirectional Transformer block
        self.transformer = TransformerBlock(d_model=256, nhead=4)

        # Pooling for variable-length to fixed-length embedding
        self.pooling = nn.AdaptiveAvgPool1d(1)

        # Final linear layer for speaker embedding z_e
        self.embedding = nn.Linear(256, embedding_dim)

    def forward(self, x):
        """
        x: (batch_size, time_steps, mel_features)
        Output: speaker embedding (batch_size, embedding_dim)
        """
        x = x.transpose(1, 2)  # (B, mel_features, T)
        x = self.encoder(x)    # (B, C, T)

        x = x.transpose(1, 2)  # (B, T, C) for transformer
        x = self.transformer(x)

        x = x.transpose(1, 2)  # (B, C, T) for pooling
        x = self.pooling(x).squeeze(-1)  # (B, C)

        embedding = self.embedding(x)  # (B, embedding_dim)
        return embedding


# Reconstruction logic (mask-guided, overlap-add placeholders)
def reconstruct_signals(masked_features, decoder):
    """
    Approximate:
        ŝ_t ≈ s_t      (target speaker)
        ŝ_r ≈ ∑ s_i + n  (residual, other speakers + noise)
    """
    reconstructed = decoder(masked_features)
    return reconstructed
