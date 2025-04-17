# models/marble_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeChannelSeparableConv1D(nn.Module):
    """
    Performs time-channel separable 1D convolution.
    First applies depthwise (temporal) convolution, then pointwise (channel) convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(TimeChannelSeparableConv1D, self).__init__()
        self.depthwise = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # Depthwise
            bias=False
        )
        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,  # Pointwise
            bias=False
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ResidualBlock(nn.Module):
    """
    A residual block with time-channel separable conv + BatchNorm + ReLU.
    """
    def __init__(self, channels, kernel_size=3, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.conv = TimeChannelSeparableConv1D(channels, channels, kernel_size=kernel_size)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)  # Equation (5): Y = ReLU(X)
        out = self.dropout(out)
        return residual + out  # Equation (4): Y = X + ResidualBlock(X)

class MarbleNet(nn.Module):
    """
    Marble-Net for VAD or Speaker Feature Extraction.
    Input: (B, C, T) - typically audio spectrograms with C channels (e.g. mel bands)
    Output: (B, F, T) - feature maps
    """
    def __init__(self, input_channels=64, num_blocks=6, dropout=0.2):
        super(MarbleNet, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(128, dropout=dropout) for _ in range(num_blocks)]
        )

        self.output_conv = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        x = self.output_conv(x)
        return x
