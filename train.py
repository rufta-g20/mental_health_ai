import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import soundfile as sf

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    Wav2Vec2FeatureExtractor, # Replace Wav2Vec2Processor
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ----------------------------------------------------------------------
# 1. Configuration & Label Mapping
# ----------------------------------------------------------------------
MODEL_CHECKPOINT = "facebook/wav2vec2-large-xlsr-53"
DATA_PATH = "data/metadata.csv"
OUTPUT_DIR = "./outputs/xlsr-ser"
TARGET_SAMPLING_RATE = 16000

# Standardized emotion classes across CREMA-D and ASED
LABELS = ["anger", "fear", "happiness", "neutral", "sadness"]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}

# ----------------------------------------------------------------------
# 2. Data Collator for Dynamic Padding
# ----------------------------------------------------------------------
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2FeatureExtractor # Changed from Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_values = [{"input_values": feature["input_values"]} for feature in features]
        labels = [feature["label"] for feature in features]

        batch = self.processor.pad(
            input_values,
            padding=self.padding,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch



# ----------------------------------------------------------------------
# 3. Audio Loading & Preprocessing
# ----------------------------------------------------------------------
def load_and_resample_audio(file_path: str):
    # Load raw audio using soundfile to prevent torchcodec/FFmpeg DLL errors on Windows
    speech_array, sr = sf.read(file_path)
    
    # Convert numpy array to PyTorch tensor -> shape: (channels, samples)
    waveform = torch.from_numpy(speech_array).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
        waveform = waveform.T

    # Convert stereo to mono if necessary
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to 16kHz required by Wav2Vec2 / XLS-R
    if sr != TARGET_SAMPLING_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLING_RATE)
        waveform = resampler(waveform)

    return waveform.squeeze().numpy()

def preprocess_function(batch, processor):
    audio_arrays = [load_and_resample_audio(path) for path in batch["file_path"]]
    inputs = processor(
        audio_arrays, 
        sampling_rate=TARGET_SAMPLING_RATE, 
        padding=False, 
        truncation=True, 
        max_length=160000 # Max 10 seconds of audio
    )
    inputs["label"] = [label2id[l] for l in batch["label"]]
    return inputs

# ----------------------------------------------------------------------
# 4. Evaluation Metrics Calculation
# ----------------------------------------------------------------------
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# ----------------------------------------------------------------------
# 5. Main Execution Flow
# ----------------------------------------------------------------------
def main():
    # Load dataset index
    df = pd.read_csv(DATA_PATH)
    # Filter rows to include only chosen standardized labels
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)

    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    # Initialize Feature Extractor
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_CHECKPOINT)

    # Process audio samples into model input formats
    print("Preprocessing training dataset...")
    train_dataset = train_dataset.map(
        lambda b: preprocess_function(b, processor),
        batched=True,
        batch_size=16,
        remove_columns=train_dataset.column_names
    )
    print("Preprocessing evaluation dataset...")
    eval_dataset = eval_dataset.map(
        lambda b: preprocess_function(b, processor),
        batched=True,
        batch_size=16,
        remove_columns=eval_dataset.column_names
    )

    # Load pre-trained XLS-R with Sequence Classification Head
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(LABELS),
        label2id=label2id,
        id2label=id2label,
    )

    # Freeze feature encoder layers to save VRAM and avoid overfitting
    model.freeze_feature_encoder()

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # Define training hyper-parameters
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        num_train_epochs=5,
        fp16=torch.cuda.is_available(),  # Mixed precision GPU acceleration
        save_steps=200,
        eval_steps=200,
        logging_steps=50,
        learning_rate=3e-5,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        push_to_hub=False,
    )

    # Instantiate Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Start Fine-Tuning
    print("Starting training...")
    trainer.train()

    # Save final model and processor
    model.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
    processor.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
    print("Training complete! Model saved to outputs/xlsr-ser/best_model.")

if __name__ == "__main__":
    main()