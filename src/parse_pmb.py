import os
import json
from pathlib import Path

def collect_pmb_data_sbn(pmb_dir, subset="gold"):
    """
    Recursively collect (sentence, sbn) pairs from the PMB 5.1.0 structure.
    We look for 'en.raw' (English) and 'en.drs.sbn' (logical form).
    
    :param pmb_dir: Path to something like '/home/everett/Desktop/parser/pmb-5.1.0/data/en'
    :param subset: 'gold' or 'silver' or 'bronze', etc., depending on your folder naming
    :return: list of dict: [{'sentence': ..., 'drs': ...}, ...]
    """
    dataset = []
    subset_path = Path(pmb_dir) / subset

    for root, dirs, files in os.walk(subset_path):
        files_set = set(files)
        # We check if both 'en.raw' and 'en.drs.sbn' are present
        if 'en.raw' in files_set and 'en.drs.sbn' in files_set:
            raw_path = Path(root) / 'en.raw'
            sbn_path = Path(root) / 'en.drs.sbn'
            
            with open(raw_path, 'r', encoding='utf-8') as f:
                english_text = f.read().strip()
            
            with open(sbn_path, 'r', encoding='utf-8') as f:
                sbn_text = f.read().strip()
            
            if english_text and sbn_text:
                dataset.append({
                    'sentence': english_text,
                    'drs': sbn_text  # We'll store the SBN as 'drs' for convenience
                })
    return dataset


if __name__ == "__main__":
    # Example usage:
    pmb_root = "/home/everett/Desktop/parser/pmb-5.1.0/data/en"
    subset = "gold"  # or 'silver'/'bronze' if you have them
    data = collect_pmb_data_sbn(pmb_root, subset)
    
    print(f"Collected {len(data)} (sentence, SBN) pairs from {subset} set.")

    import os

    out_dir = os.path.join("..", "data")  # ../data
    out_file = "pmb_gold_sbn.json"

    os.makedirs(out_dir, exist_ok=True)   # Create the folder if it doesn't exist

    file_path = os.path.join(out_dir, out_file)
    with open(file_path, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, indent=2, ensure_ascii=False)
    print(f"Saved data to {out_file}")
