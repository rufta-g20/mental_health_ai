import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

def translate_dataset(
    input_csv="data/raw/text/Sentiment_Mental_health_dataset.csv",
    output_csv="data/processed/synthetic_amharic_text.csv",
    batch_size=16,
    sample_size=None
):
    print("--- Starting English to Amharic Translation Pipeline ---")
    
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found at: {input_csv}")
        
    df = pd.read_csv(input_csv)
    print(f"Loaded dataset with {len(df)} rows.")
    
    # Identify key columns dynamically
    text_col = 'Statement' if 'Statement' in df.columns else df.columns[0]
    status_col = 'Status' if 'Status' in df.columns else df.columns[1]
    
    # Clean text inputs
    df[text_col] = df[text_col].astype(str).fillna("")
    
    if sample_size:
        df = df.head(sample_size).copy()
        print(f"Sampled first {sample_size} rows for testing.")

    # Determine CPU or CUDA Hardware Acceleration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device.upper()}")

    model_name = "facebook/nllb-200-distilled-600M"
    print(f"Loading model '{model_name}'...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    # NLLB Language Codes: English (eng_Latn), Amharic (amh_Ethi)
    src_lang = "eng_Latn"
    tgt_lang = "amh_Ethi"
    
    tokenizer.src_lang = src_lang
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    translated_texts = []
    statements = df[text_col].tolist()

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    print("Translating text batches...")
    for i in tqdm(range(0, len(statements), batch_size)):
        batch_text = statements[i : i + batch_size]
        
        try:
            inputs = tokenizer(
                batch_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(device)

            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_lang_id,
                    max_length=256
                )

            batch_translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            translated_texts.extend(batch_translations)

        except Exception as e:
            print(f"\n[Warning] Batch error at index {i}: {e}. Fallback to original text.")
            translated_texts.extend(batch_text)

        # Save checkpoint every 500 rows
        if (i + batch_size) % 500 < batch_size:
            temp_df = df.iloc[: len(translated_texts)].copy()
            temp_df["Statement_Amharic"] = translated_texts
            temp_df.to_csv(output_csv, index=False)

    df["Statement_Amharic"] = translated_texts
    df.to_csv(output_csv, index=False)
    print(f"\nSUCCESS! Translated dataset saved to: {output_csv}")

if __name__ == "__main__":
    # Change sample_size=None to process the full dataset
    translate_dataset(batch_size=16, sample_size=None)