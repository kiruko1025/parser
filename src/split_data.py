#!/usr/bin/env python3

import json
import random
import os

def split_dataset(json_file, train_ratio=0.8, val_ratio=0.1):
    """
    Splits a JSON dataset into train, val, and test sets.
    json_file: path to the cleaned JSON file (each entry has 'sentence', 'drs', etc.).
    """
    if not os.path.exists(json_file):
        print(f"Error: {json_file} does not exist.")
        return [], [], []

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    random.shuffle(data)
    total = len(data)
    train_end = int(total * train_ratio)
    val_end   = int(total * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data   = data[train_end:val_end]
    test_data  = data[val_end:]

    return train_data, val_data, test_data


if __name__ == "__main__":
    # Base directory for data (one level up from 'src/')
    base_dir = os.path.join("..", "data")

    # Input file name inside ../data
    input_file_name = "pmb_gold_sbn_cleaned.json"
    input_file_path = os.path.join(base_dir, input_file_name)

    # Perform the split
    train_data, val_data, test_data = split_dataset(input_file_path)

    print(f"Train: {len(train_data)}")
    print(f"Val:   {len(val_data)}")
    print(f"Test:  {len(test_data)}")

    # Save splits back to the data folder
    train_out_path = os.path.join(base_dir, "train.json")
    val_out_path   = os.path.join(base_dir, "val.json")
    test_out_path  = os.path.join(base_dir, "test.json")

    with open(train_out_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open(val_out_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    with open(test_out_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print(f"Train/Val/Test splits saved to {os.path.abspath(base_dir)}")

