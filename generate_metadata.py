import os
import re
import pandas as pd

# Define target paths for your downloaded datasets
CREMA_DIR = "data/raw/audio/crema_d" # Adjust to your CREMA-D wav directory
ASED_DIR = "data/raw/audio/ased"        # Adjust to your ASED wav directory
OUTPUT_CSV = "data/metadata.csv"

# ----------------------------------------------------------------------
# 1. Standardized Emotion Mapping
# ----------------------------------------------------------------------
# Target labels: ["anger", "fear", "happiness", "neutral", "sadness"]

# CREMA-D emotion codes (e.g., 1001_DFA_ANG_XX.wav)
CREMA_MAP = {
    "ANG": "anger",
    "FEA": "fear",
    "HAP": "happiness",
    "NEU": "neutral",
    "SAD": "sadness"
    # "DIS": "disgust" is excluded or ignored to align 5-class target
}

# ASED emotion prefixes (e.g., a5-03-01-02-50.wav)
# ASED prefixes start with the first letter of the emotion (a=anger, f=fear, h=happy, n=neutral, s=sad)
ASED_MAP = {
    "a": "anger",
    "f": "fear",
    "h": "happiness",
    "n": "neutral",
    "s": "sadness"
}

data = []

# ----------------------------------------------------------------------
# 2. Parse CREMA-D Files
# ----------------------------------------------------------------------
if os.path.exists(CREMA_DIR):
    print(f"Processing CREMA-D files from: {CREMA_DIR}")
    for root, _, files in os.walk(CREMA_DIR):
        for file in files:
            if file.endswith(".wav"):
                # Filename pattern: ActorID_Sentence_Emotion_EmotionLevel.wav
                parts = file.split("_")
                if len(parts) >= 3:
                    emo_code = parts[2]
                    if emo_code in CREMA_MAP:
                        full_path = os.path.join(root, file)
                        data.append({
                            "file_path": full_path,
                            "label": CREMA_MAP[emo_code],
                            "dataset": "CREMA-D"
                        })
else:
    print(f"Warning: CREMA-D directory '{CREMA_DIR}' not found. Check your path.")

# ----------------------------------------------------------------------
# 3. Parse ASED Files
# ----------------------------------------------------------------------
if os.path.exists(ASED_DIR):
    print(f"Processing ASED files from: {ASED_DIR}")
    for root, _, files in os.walk(ASED_DIR):
        for file in files:
            if file.endswith(".wav"):
                # Filename pattern: xn-yy-aa-gg-bb.wav
                # e.g., 'a5-03-01-02-50.wav' -> prefix is 'a'
                filename = os.path.basename(file).lower()
                prefix_letter = filename[0] if filename else ""
                
                if prefix_letter in ASED_MAP:
                    full_path = os.path.join(root, file)
                    data.append({
                        "file_path": full_path,
                        "label": ASED_MAP[prefix_letter],
                        "dataset": "ASED"
                    })
else:
    print(f"Warning: ASED directory '{ASED_DIR}' not found. Check your path.")

# ----------------------------------------------------------------------
# 4. Save Metadata & Display Statistics
# ----------------------------------------------------------------------
df = pd.DataFrame(data)

if not df.empty:
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # Save required columns to CSV
    df[["file_path", "label"]].to_csv(OUTPUT_CSV, index=False)
    
    print(f"\nSuccessfully generated '{OUTPUT_CSV}' with {len(df)} entries.")
    print("\n--- Summary Breakdown by Dataset and Emotion ---")
    print(pd.crosstab(df['dataset'], df['label']))
else:
    print("\nNo matching audio files found. Please verify your folder paths.")