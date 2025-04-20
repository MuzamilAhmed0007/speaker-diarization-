# models/neural_diarizer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TDNNBlock(nn.Module):
    """
    Time Delay Neural Network (TDNN) layer for learning speaker embeddings.
    Equation: h^(i) = ReLU(W^i h^(i-1) + b^i)
    """
    def __init__(self, input_dim, output_dim, context_size=5, dilation=1):
        super(TDNNBlock, self).__init__()
        self.context_size = context_size
        self.conv1d = nn.Conv1d(input_dim, output_dim,
                                kernel_size=context_size,
                                dilation=dilation)

    def forward(self, x):
        x = self.conv1d(x)
        x = F.relu(x)
        return x


class CustomLSTMCell(nn.Module):
    """
    LSTM cell implementation using equations (13)-(18)
    """
    def __init__(self, input_dim, hidden_dim):
        super(CustomLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.V_f = nn.Linear(input_dim, hidden_dim)
        self.M_f = nn.Linear(hidden_dim, hidden_dim)
        self.b_f = nn.Parameter(torch.zeros(hidden_dim))

        self.V_i = nn.Linear(input_dim, hidden_dim)
        self.M_i = nn.Linear(hidden_dim, hidden_dim)
        self.b_i = nn.Parameter(torch.zeros(hidden_dim))

        self.V_o = nn.Linear(input_dim, hidden_dim)
        self.M_o = nn.Linear(hidden_dim, hidden_dim)
        self.b_o = nn.Parameter(torch.zeros(hidden_dim))

        self.V_c = nn.Linear(input_dim, hidden_dim)
        self.M_c = nn.Linear(hidden_dim, hidden_dim)
        self.b_c = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, e_t, h_prev, c_prev):
        g_t = torch.sigmoid(self.V_f(e_t) + self.M_f(h_prev) + self.b_f)
        p_t = torch.sigmoid(self.V_i(e_t) + self.M_i(h_prev) + self.b_i)
        q_t = torch.sigmoid(self.V_o(e_t) + self.M_o(h_prev) + self.b_o)
        m_t_hat = torch.tanh(self.V_c(e_t) + self.M_c(h_prev) + self.b_c)
        c_t = g_t * c_prev + p_t * m_t_hat
        h_t = q_t * torch.tanh(c_t)
        return h_t, c_t


class NeuralDiarizer(nn.Module):
    """
    Neural-TM Diarizer with hybrid TDNN and LSTM for speaker diarization.
    Outputs speaker label per segment.
    """
    def __init__(self, input_dim=192, tdnn_hidden=256, lstm_hidden=128, num_speakers=10):
        super(NeuralDiarizer, self).__init__()

        # TDNN layers for learning speaker-invariant features
        self.tdnn = nn.Sequential(
            TDNNBlock(input_dim, tdnn_hidden, context_size=5),
            TDNNBlock(tdnn_hidden, tdnn_hidden, context_size=3),
            TDNNBlock(tdnn_hidden, tdnn_hidden, context_size=1),
        )

        # LSTM cell for sequential modeling
        self.lstm_cell = CustomLSTMCell(tdnn_hidden, lstm_hidden)

        # Final classification layer (eq. 19)
        self.classifier = nn.Linear(lstm_hidden, num_speakers)

    def forward(self, segments):
        """
        segments: Tensor of shape (batch, time, features) where time is segment count
        """
        batch_size, time_steps, feat_dim = segments.size()
        x = segments.transpose(1, 2)  # (batch, features, time)

        tdnn_out = self.tdnn(x)  # (batch, hidden_dim, time)
        tdnn_out = tdnn_out.transpose(1, 2)  # (batch, time, hidden_dim)

        h_t = torch.zeros(batch_size, self.lstm_cell.hidden_dim, device=segments.device)
        c_t = torch.zeros(batch_size, self.lstm_cell.hidden_dim, device=segments.device)

        outputs = []
        for t in range(tdnn_out.size(1)):
            e_t = tdnn_out[:, t, :]  # Current segment's TDNN feature
            h_t, c_t = self.lstm_cell(e_t, h_t, c_t)
            outputs.append(h_t)

        h_seq = torch.stack(outputs, dim=1)  # (batch, time, hidden_dim)
        y_hat = self.classifier(h_seq)       # (batch, time, num_speakers)

        return F.log_softmax(y_hat, dim=-1)
