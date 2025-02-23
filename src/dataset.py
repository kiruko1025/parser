import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer

class PmbSbnDataset(Dataset):
    def __init__(self, data_list, tokenizer, source_max_len=128, target_max_len=256):
        """
        data_list: a list of dicts with "sentence" and "drs" keys
        tokenizer: e.g. a T5Tokenizer
        """
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        sentence = item['sentence']
        sbn = item['drs']

        # Provide a prefix for clarity: "translate English to SBN:"
        source_text = f"translate English to SBN: {sentence}"
        target_text = sbn  # The SBN is our target

        # Tokenize the source
        source_encodings = self.tokenizer(
            source_text, 
            truncation=True,
            padding="max_length",
            max_length=self.source_max_len,
            return_tensors="pt"
        )
        # Tokenize the target
        target_encodings = self.tokenizer(
            target_text, 
            truncation=True,
            padding="max_length",
            max_length=self.target_max_len,
            return_tensors="pt"
        )

        input_ids = source_encodings["input_ids"].squeeze()
        attention_mask = source_encodings["attention_mask"].squeeze()
        labels = target_encodings["input_ids"].squeeze()

        # T5 uses -100 for ignored positions in the labels (where padding occurs)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }



model_checkpoint = "t5-small"  # Could switch to "t5-base", "t5-large", "google/flan-t5-base", etc.
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)

# We'll define a prefix for clarity:
prefix = "translate English to SBN: "

# Max lengths (tweak if needed):
source_max_length = 128
target_max_length = 256

def preprocess_function(examples):
    """
    Tokenizes the English sentence (with a prefix) and the SBN string.
    """
    inputs  = [prefix + s for s in examples["sentence"]]
    targets = examples["drs"]

    model_inputs = tokenizer(
        inputs,
        max_length=source_max_length,
        padding="max_length",
        truncation=True
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=target_max_length,
            padding="max_length",
            truncation=True
        )
    
    # T5 expects padding tokens in labels to be replaced with -100
    labels["input_ids"] = [
        [(-100 if token == tokenizer.pad_token_id else token) for token in l]
        for l in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["decoder_attention_mask"] = labels["attention_mask"]
    return model_inputs