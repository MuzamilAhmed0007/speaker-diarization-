# main.py

import os
import torch
from torch.utils.data import DataLoader
from models.neural_diarizer import NeuralDiarizer
from models.titanet import TitaNet
from models.marbelnet import MarbelNet
from utils.preprocessing import preprocess_audio
from utils.segmentation import segment_audio
from utils.eval import compute_metrics, print_metrics
from utils.dataloader import DiarizationDataset

def run_pipeline(audio_path, model_paths, device):
    # === Step 1: Audio Preprocessing ===
    print("Step 1: Preprocessing audio...")
    waveform, sample_rate = preprocess_audio(audio_path)

    # === Step 2: VAD using MarbelNet ===
    print("Step 2: Voice Activity Detection using MarbelNet...")
    vad_model = MarbelNet().to(device)
    vad_model.load_state_dict(torch.load(model_paths['marbelnet']))
    speech_segments = vad_model.detect_speech(waveform, sample_rate)

    # === Step 3: Segmentation using Hybrid BIC + GMM ===
    print("Step 3: Segmenting audio into speaker-homogeneous regions...")
    segments = segment_audio(waveform, speech_segments, sample_rate)

    # === Step 4: Speaker Embedding Extraction using TitaNet ===
    print("Step 4: Extracting speaker embeddings with TitaNet...")
    titan_model = TitaNet().to(device)
    titan_model.load_state_dict(torch.load(model_paths['titanet']))
    embeddings = titan_model.extract_embeddings(segments, sample_rate)

    # === Step 5: Neural Diarization ===
    print("Step 5: Diarizing with TDNN + LSTM model...")
    diarizer = NeuralDiarizer(input_dim=embeddings.shape[-1]).to(device)
    diarizer.load_state_dict(torch.load(model_paths['neural_diarizer']))

    diarizer.eval()
    with torch.no_grad():
        outputs = diarizer(embeddings.unsqueeze(0).to(device))  # add batch dim

    return outputs


def main():
    # === Configurations ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_paths = {
        "marbelnet": "models/marbelnet/marbelnet.pt",
        "titanet": "models/titanet/titanet.pt",
        "neural_diarizer": "models/neural_diarizer/neural_diarizer.pt"
    }

    # === Dataset paths ===
    test_audio_dir = "data/test_audio"
    ground_truth_dir = "data/test_labels"

    # === Loop through test audios ===
    all_preds, all_labels = [], []

    for filename in os.listdir(test_audio_dir):
        if not filename.endswith(".wav"):
            continue

        audio_path = os.path.join(test_audio_dir, filename)
        label_path = os.path.join(ground_truth_dir, filename.replace(".wav", ".pt"))

        print(f"\n--- Processing {filename} ---")
        pred = run_pipeline(audio_path, model_paths, device)
        label = torch.load(label_path)

        all_preds.append(pred.squeeze(0))  # remove batch dim
        all_labels.append(label)

    # === Evaluate ===
    print("\n=== Evaluation ===")
    for pred, label in zip(all_preds, all_labels):
        der, fa, md = compute_metrics(pred, label)
        print_metrics(der, fa, md)


if __name__ == "__main__":
    main()
