# models/neural_diarizer.py

import torch
import torch.nn as nn

class NeuralDiarizer(nn.Module):
    def __init__(self, input_size=256, hidden_size=128, num_classes=10):
        super(NeuralDiarizer, self).__init__()
        self.tdnn = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.tdnn(x.transpose(1, 2))
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        return self.output(lstm_out)
