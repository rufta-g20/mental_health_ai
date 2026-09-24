import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight

def prepare_data_splits(input_csv, output_dir):
    print(f"Loading cleaned dataset: {input_csv}")
    df = pd.read_csv(input_csv)

    # 1. Encode Target Labels
    label_encoder = LabelEncoder()
    df["target"] = label_encoder.fit_transform(df["status"])
    
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.pkl"))

    # 2. First Split: 80% Train, 20% Temp (Val + Test)
    X_train_df, X_temp_df, y_train, y_temp = train_test_split(
        df,
        df["target"],
        test_size=0.20,
        random_state=42,
        stratify=df["target"]
    )

    # 3. Second Split: Divide 20% Temp evenly into 10% Validation and 10% Test
    X_val_df, X_test_df, y_val, y_test = train_test_split(
        X_temp_df,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print(f"\n--- Split Distribution Complete ---")
    print(f"Train Set Shape      : {X_train_df.shape} ({len(X_train_df)/len(df):.0%})")
    print(f"Validation Set Shape : {X_val_df.shape} ({len(X_val_df)/len(df):.0%})")
    print(f"Test Set Shape       : {X_test_df.shape} ({len(X_test_df)/len(df):.0%})")

    # 4. Feature Engineering (TF-IDF Vectorization)
    print("\nExtracting TF-IDF Features...")
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    
    # Fit ONLY on Training data to prevent information leakage
    X_train_tfidf = tfidf.fit_transform(X_train_df["cleaned_english"])
    X_val_tfidf = tfidf.transform(X_val_df["cleaned_english"])
    X_test_tfidf = tfidf.transform(X_test_df["cleaned_english"])

    # Save Vectorizer
    joblib.dump(tfidf, os.path.join(output_dir, "tfidf_vectorizer.pkl"))

    # 5. Compute Class Weights for Handling Imbalance
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    print(f"Computed Class Weights (Balanced): {class_weight_dict}")

    # Save Splits to CSV for Reproducibility
    X_train_df.to_csv(os.path.join(output_dir, "train_split.csv"), index=False)
    X_val_df.to_csv(os.path.join(output_dir, "val_split.csv"), index=False)
    X_test_df.to_csv(os.path.join(output_dir, "test_split.csv"), index=False)
    
    print(f"\nAll splits and feature artifacts saved to: {output_dir}")

if __name__ == "__main__":
    prepare_data_splits(
        input_csv="data/processed/cleaned_multilingual_text.csv",
        output_dir="data/processed/splits/"
    )