import os
import glob
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# CREMA-D Filename Code Mapping
CREMA_EMOTION_MAP = {
    'SAD': 'Sadness',
    'ANG': 'Anger',
    'DIS': 'Disgust',
    'FEA': 'Fear',
    'HAP': 'Happiness',
    'NEU': 'Neutral'
}

# Expanded Mapping for ASED Codes based on Filename Prefix Patterns (e.g., 'a5-03-01-02-50.wav')
ASED_EMOTION_MAP = {
    'a': 'Anger',
    'h': 'Happiness',
    's': 'Sadness',
    'f': 'Fear',
    'd': 'Disgust',
    'n': 'Neutral'
}

def extract_signal_features(file_path):
    """Extract acoustic biomarkers from raw audio using Librosa."""
    try:
        # Load audio at 16kHz sampling rate
        y, sr = librosa.load(file_path, sr=16000, duration=3.0)
        
        if len(y) == 0:
            return None

        # 1. Zero Crossing Rate (ZCR)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        
        # 2. Root Mean Square Energy (RMSE)
        rmse = np.mean(librosa.feature.rmse(y=y) if hasattr(librosa.feature, 'rmse') else librosa.feature.rms(y=y))
        
        # 3. Spectral Centroid & Rolloff
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        # 4. Pitch (F0 Estimate via Harmonic Representation)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0.0

        # 5. Mel-Frequency Cepstral Coefficients (20 MFCCs)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_means = np.mean(mfccs.T, axis=0)

        # Combine feature vector
        features = [zcr, rmse, spec_centroid, spec_rolloff, pitch_mean]
        features.extend(mfcc_means)
        
        return features

    except Exception as e:
        return None

def process_audio_directory(dataset_type, input_dir, output_csv):
    print(f"\n--- Processing {dataset_type.upper()} Audio Dataset ---")
    
    audio_files = glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True)
    print(f"Found {len(audio_files)} .wav audio files in {input_dir}.")

    if len(audio_files) == 0:
        print(f"[Skipped] No .wav files found in {input_dir}. Check file extraction.")
        return

    records = []
    
    # Feature Header Names
    columns = ["filename", "dataset", "label", "zcr", "rmse", "spectral_centroid", "spectral_rolloff", "pitch_mean"]
    columns.extend([f"mfcc_{i+1}" for i in range(20)])

    for filepath in tqdm(audio_files):
        filename = os.path.basename(filepath)
        label = "Unknown"

        # Parsing Labels based on Dataset Structure
        if dataset_type == "crema_d":
            parts = filename.replace('.wav', '').split('_')
            if len(parts) >= 3:
                emotion_code = parts[2]
                label = CREMA_EMOTION_MAP.get(emotion_code, "Unknown")
                
        elif dataset_type == "ased":
            # Extract prefix prior to hyphen (e.g., 'a5' from 'a5-03-01-02-50.wav')
            code_prefix = filename.split('-')[0].lower()
            emotion_char = code_prefix[0] if len(code_prefix) > 0 else ''
            label = ASED_EMOTION_MAP.get(emotion_char, "Unknown")

        feature_vector = extract_signal_features(filepath)
        
        if feature_vector is not None:
            row = [filename, dataset_type, label] + feature_vector
            records.append(row)

    df = pd.DataFrame(records, columns=columns)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} extracted feature vectors to: {output_csv}")

if __name__ == "__main__":
    # Extract CREMA-D Features
    process_audio_directory(
        dataset_type="crema_d",
        input_dir="data/raw/audio/crema_d",
        output_csv="data/processed/crema_d_audio_features.csv"
    )
    
    # Extract ASED Features
    process_audio_directory(
        dataset_type="ased",
        input_dir="data/raw/audio/ased",
        output_csv="data/processed/ased_audio_features.csv"
    )