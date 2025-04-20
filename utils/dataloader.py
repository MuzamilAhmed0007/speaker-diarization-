# utils/dataloader.py

import os
import torch
from torch.utils.data import Dataset
from torchaudio import load

class DiarizationDataset(Dataset):
    """
    Custom Dataset for loading diarization audio and labels
    """

    def __init__(self, audio_dir, label_dir, transform=None):
        self.audio_dir = audio_dir
        self.label_dir = label_dir
        self.transform = transform
        self.filenames = [
            f for f in os.listdir(audio_dir) if f.endswith(".wav")
        ]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        audio_path = os.path.join(self.audio_dir, fname)
        label_path = os.path.join(self.label_dir, fname.replace(".wav", ".pt"))

        waveform, sr = load(audio_path)
        label = torch.load(label_path)

        if self.transform:
            waveform = self.transform(waveform)

        return {
            "waveform": waveform,
            "sample_rate": sr,
            "label": label,
            "filename": fname
        }
