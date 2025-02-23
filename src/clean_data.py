#!/usr/bin/env python3

"""
clean_data.py

A script to clean SBN/DRS data in a JSON file by:
1. Removing lines that start with '%%%'.
2. Removing ANSI color codes.

Usage:
    python clean_data.py

It will read from 'pmb_gold_sbn.json' and produce 'pmb_gold_sbn_cleaned.json'.
Adjust filenames as desired in the script or via command line modifications.
"""

import re
import json
import os

# Regex to detect ANSI escape codes, e.g. \u001b[31m
ANSI_ESCAPE_PATTERN = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

def clean_sbn_text(raw_sbn: str) -> str:
    """
    Remove lines starting with '%%%' (Boxer commands) and
    strip out ANSI escape sequences.

    :param raw_sbn: The raw SBN/DRS string from the PMB file.
    :return: Cleaned SBN string.
    """
    lines = raw_sbn.split('\n')
    # Filter out lines that start with '%%%'
    filtered_lines = [ln for ln in lines if not ln.strip().startswith('%%%')]
    # Remove ANSI escape codes
    cleaned_lines = [ANSI_ESCAPE_PATTERN.sub('', ln) for ln in filtered_lines]
    # Re-join the cleaned lines
    cleaned_sbn = '\n'.join(cleaned_lines).strip()
    return cleaned_sbn

def clean_data(input_file: str, output_file: str):
    """
    Load JSON data from input_file, clean each 'drs' field,
    then save the result to output_file.

    :param input_file: Path to the JSON file to clean.
    :param output_file: Path to the new JSON file to be created.
    """
    if not os.path.exists(input_file):
        print(f"Error: {input_file} does not exist.")
        return
    
    # Load the data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned_data = []
    for item in data:
        sentence = item.get('sentence', '')
        raw_sbn = item.get('drs', '')
        
        cleaned_sbn = clean_sbn_text(raw_sbn)
        
        cleaned_data.append({
            'sentence': sentence,
            'drs': cleaned_sbn
        })
    
    # Save cleaned data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    
    print(f"Done! Cleaned data saved to {output_file}.")
    print(f"Processed {len(data)} entries.")

if __name__ == "__main__":
    # You can change these defaults or make them command-line arguments.

    base_dir = os.path.join("..", "data")

    # 2) Input file name
    input_file_name = "pmb_gold_sbn.json"

    # 3) Output file name
    output_file_name = "pmb_gold_sbn_cleaned.json"

    # 4) Combine to get full paths
    input_file_path  = os.path.join(base_dir, input_file_name)
    output_file_path = os.path.join(base_dir, output_file_name)

    # 5) Run the cleaning function
    clean_data(input_file_path, output_file_path)

