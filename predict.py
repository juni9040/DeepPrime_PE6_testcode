"""
DeepPrime PE6 inference script.

Accepts two input modes:
  (A) Raw input  — CSV with pegRNA design parameters (sequences + edit info).
                   Bio-features are computed automatically.
  (B) Pre-processed — CSV that already contains bio-feature columns (Tm1_PBS, etc.).
                   Preprocessing is skipped.

Detection: if 'Tm1_PBS' column is present → mode B, otherwise → mode A.

Usage (mode A – raw input):
    python predict.py \\
        --train_csv test_input.csv \\
        --val_csv   test_input.csv \\
        --model_ckpt ckpt/drn.ckpt \\
        --output_csv predictions.csv

Usage (mode B – pre-processed):
    python predict.py \\
        --train_csv rq3_train.csv \\
        --val_csv   data/dp_pe4max_reformat.csv \\
        --model_ckpt ckpt/drn.ckpt \\
        --output_csv predictions.csv

Required columns for raw input CSV (mode A):
    REF_ID, WideTargetSequence, OligoSequence_fixed_length, Guide,
    PBS, RTT, Edit_type, Edit_len, Edit_pos, PBS_len, RTT_len,
    leading G, <pe_type> (e.g. PEmaxdRNaseH), read-count columns
    (see test_input.csv for a complete example)
"""
import argparse
import sys
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from model import EnsembleModel
from dataset import PE6DeepPrimeDataset


def parse_args():
    parser = argparse.ArgumentParser(description="DeepPrime PE6 Inference")
    parser.add_argument("--train_csv", type=str, required=True,
                        help="Training CSV (used for normalization statistics)")
    parser.add_argument("--val_csv", type=str, required=True,
                        help="Validation/test CSV to run predictions on")
    parser.add_argument("--model_ckpt", type=str, required=True,
                        help="Path to .ckpt checkpoint file")
    parser.add_argument("--output_csv", type=str, default="predictions.csv",
                        help="Path to save output predictions")
    parser.add_argument("--pe_type", type=str, default="PEmaxdRNaseH",
                        help="PE type column name in the CSV (default: PEmaxdRNaseH)")
    parser.add_argument("--seq_len", type=int, default=100,
                        help="Sequence length for padding (default: 100)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def fix_seq(s, length):
    s = str(s)
    return s[:length] if len(s) > length else s + "X" * (length - len(s))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonize column name variants."""
    mapping = {
        "PBSlen": "PBS_len",
        "RTlen": "RTT_len",
        "RT_len": "RTT_len",
        "RT-PBSlen": "RT-PBS_len",
    }
    for old, new in mapping.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    if "No." in df.columns and "ID" not in df.columns:
        df["ID"] = df["No."]
    elif "REF_ID" in df.columns and "ID" not in df.columns:
        df["ID"] = df["REF_ID"]
    return df


def has_biofeatures(df: pd.DataFrame) -> bool:
    return "Tm1_PBS" in df.columns and "Target" in df.columns


def load_and_preprocess(path: str) -> pd.DataFrame:
    """Load CSV and compute bio-features if not already present."""
    df = pd.read_csv(path)
    if has_biofeatures(df):
        print(f"  Bio-features detected in {path} — skipping preprocessing.")
        return normalize_columns(df)

    print(f"  No bio-features found in {path} — running preprocessing (requires genet, biopython, ViennaRNA)...")
    from preprocess import preprocess_data
    df = preprocess_data(df)
    df = normalize_columns(df)
    print(f"  Preprocessing complete. {len(df)} rows processed.")
    return df


def main():
    args = parse_args()
    print(f"Device: {args.device}")

    print("Loading train CSV...")
    df_train = load_and_preprocess(args.train_csv)

    print("Loading val CSV...")
    df_val = load_and_preprocess(args.val_csv)

    # Pad sequences to fixed length
    for df in [df_train, df_val]:
        df["Target"] = df["Target"].apply(lambda x: fix_seq(x, args.seq_len))
        df["Masked_EditSeq"] = df["Masked_EditSeq"].apply(lambda x: fix_seq(x, args.seq_len))

    datafilter = {"PE_types": [args.pe_type]}

    # Build train dataset to get normalization stats
    train_dataset = PE6DeepPrimeDataset(data=df_train, datafilter=datafilter)
    norm_mean = train_dataset.norm_mean
    norm_std = train_dataset.norm_std

    # Build val dataset
    val_dataset = PE6DeepPrimeDataset(
        data=df_val, datafilter=datafilter,
        norm_mean=norm_mean, norm_std=norm_std,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    print(f"Loading model from {args.model_ckpt}...")
    model = EnsembleModel.from_checkpoint(args.model_ckpt, device=args.device)
    print(f"Loaded ensemble of {len(model.feature_extractor.models)} models.")

    # Inference
    print("Running inference...")
    all_preds, all_targets, all_ids = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            ((g, b), y), ids = batch
            preds = model(g.to(args.device).float(), b.to(args.device).float())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_ids.extend(ids)

    result_df = pd.DataFrame({
        "id": all_ids,
        "true": np.concatenate(all_targets)[:, 0],
        "pred": np.concatenate(all_preds).flatten(),
    })
    result_df.to_csv(args.output_csv, index=False)
    print(f"Done. Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
