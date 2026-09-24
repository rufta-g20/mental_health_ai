import os
import re
import pandas as pd
import numpy as np

def clean_english_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+|\#", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_amharic_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\u1200-\u137F\s\.\,\!\?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_and_clean_dataset(input_csv, output_csv):
    print(f"Reading dataset: {input_csv}")
    df = pd.read_csv(input_csv)

    # Standardize column headers to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Clean text columns
    english_col = 'statement' if 'statement' in df.columns else df.columns[0]
    df["cleaned_english"] = df[english_col].apply(clean_english_text)

    if "statement_amharic" in df.columns:
        df["cleaned_amharic"] = df["statement_amharic"].apply(clean_amharic_text)
        df["cleaned_amharic"] = df["cleaned_amharic"].replace("", np.nan)

    # Drop empty string artifacts from cleaning
    df["cleaned_english"] = df["cleaned_english"].replace("", np.nan)
    
    # 1. Drop missing values in crucial columns
    subset_cols = ["cleaned_english", "status"]
    if "cleaned_amharic" in df.columns:
        subset_cols.append("cleaned_amharic")
    df = df.dropna(subset=subset_cols).reset_index(drop=True)

    # 2. Drop duplicates based on cleaned English text
    initial_len = len(df)
    df = df.drop_duplicates(subset=["cleaned_english"]).reset_index(drop=True)
    print(f"Removed {initial_len - len(df)} duplicate records.")

    # 3. Add Basic Tabular Feature Engineering Metrics
    df["en_word_count"] = df["cleaned_english"].apply(lambda x: len(x.split()))
    if "cleaned_amharic" in df.columns:
        df["am_word_count"] = df["cleaned_amharic"].apply(lambda x: len(str(x).split()))

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Successfully saved clean dataset to: {output_csv} (Total Rows: {len(df)})")

if __name__ == "__main__":
    raw_processed = "data/processed/synthetic_amharic_text.csv"
    if os.path.exists(raw_processed):
        preprocess_and_clean_dataset(
            input_csv=raw_processed,
            output_csv="data/processed/cleaned_multilingual_text.csv"
        )