#!/usr/bin/env python3
"""
train.py

Trains a T5 model to map English sentences to SBN (logical forms).
Assumes you have train.json and val.json (and optionally test.json)
in the same directory.

Usage:
  python train.py
  # optionally: python train.py --train_file=<path> --val_file=<path> --test_file=<path>

Requirements:
  pip install transformers datasets evaluate sentencepiece
"""

import argparse
import json
import random
import os

import evaluate
import torch
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer
)

# -------------
# Config
# -------------
DEFAULT_MODEL_CHECKPOINT = "t5-small"       # or "t5-base", "google/flan-t5-base", etc.
PREFIX = "translate English to SBN: "       # prefix for the input
SOURCE_MAX_LEN = 128
TARGET_MAX_LEN = 256
BATCH_SIZE = 4
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4


# -------------
# Functions
# -------------

def load_json_as_dataset(json_path):
    """
    Loads a JSON file with [{"sentence":..., "drs":...}, ...]
    and converts it to a HuggingFace Dataset.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} does not exist.")
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    # Convert list of dicts into a Dataset
    dataset = Dataset.from_list(data_list)
    return dataset


def preprocess_function(examples, tokenizer):
    """
    Tokenizes English sentences (with a prefix) and SBN targets.
    """
    inputs = [PREFIX + s for s in examples["sentence"]]
    targets = examples["drs"]

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=SOURCE_MAX_LEN,
        padding="max_length",
        truncation=True
    )
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=TARGET_MAX_LEN,
            padding="max_length",
            truncation=True
        )

    # Replace pad tokens in labels with -100
    labels["input_ids"] = [
        [(-100 if token == tokenizer.pad_token_id else token) for token in seq]
        for seq in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["decoder_attention_mask"] = labels["attention_mask"]
    return model_inputs


def compute_metrics_sacrebleu(eval_pred, tokenizer):
    """
    Computes BLEU score with sacreBLEU. 
    You can switch to an exact match metric if you prefer.
    """
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 with pad_token_id in the labels
    labels = [
        [token if token != -100 else tokenizer.pad_token_id for token in label]
        for label in labels
    ]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Evaluate with sacreBLEU
    bleu = evaluate.load("sacrebleu")
    # sacreBLEU wants each reference to be a list
    results = bleu.compute(
        predictions=decoded_preds, 
        references=[[label] for label in decoded_labels]
    )
    return {"bleu": results["score"]}


# -------------
# Main
# -------------
def main(args):
    # Load train and val data
    print("Loading data...")
    train_dataset = load_json_as_dataset(args.train_file)
    val_dataset = load_json_as_dataset(args.val_file)

    # Optional test dataset
    test_dataset = None
    if args.test_file and os.path.exists(args.test_file):
        test_dataset = load_json_as_dataset(args.test_file)

    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.model_checkpoint)

    # Preprocess data
    def map_fn(examples):
        return preprocess_function(examples, tokenizer)

    train_dataset = train_dataset.map(map_fn, batched=True)
    val_dataset = val_dataset.map(map_fn, batched=True)
    if test_dataset:
        test_dataset = test_dataset.map(map_fn, batched=True)

    # Convert datasets to PyTorch format
    cols = ["input_ids", "attention_mask", "labels", "decoder_attention_mask"]
    train_dataset.set_format("torch", columns=cols)
    val_dataset.set_format("torch", columns=cols)
    if test_dataset:
        test_dataset.set_format("torch", columns=cols)

    # Load model
    print(f"Loading model: {args.model_checkpoint}")
    model = T5ForConditionalGeneration.from_pretrained(args.model_checkpoint)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=100,
        load_best_model_at_end=True,
        # More advanced features could go here (lr_scheduler_type, warmup_steps, etc.)
    )

    # Define compute_metrics function
    def compute_metrics(eval_pred):
        return compute_metrics_sacrebleu(eval_pred, tokenizer)

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Evaluate on val
    print("Evaluating on validation set...")
    val_metrics = trainer.evaluate(eval_dataset=val_dataset)
    print("Validation metrics:", val_metrics)

    # If test dataset is provided, evaluate on test set
    if test_dataset:
        print("Evaluating on test set...")
        test_metrics = trainer.evaluate(eval_dataset=test_dataset)
        print("Test metrics:", test_metrics)

    # Save the final model
    print(f"Saving model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # You can build default paths to ../data using os.path.join:
    default_train_file = os.path.join("..", "data", "train.json")
    default_val_file   = os.path.join("..", "data", "val.json")
    default_test_file  = os.path.join("..", "data", "test.json")
    default_output_dir = os.path.join("..", "models", "my_t5_sbn_model")

    parser.add_argument(
        "--train_file", type=str, 
        default=default_train_file,
        help="Path to training data JSON."
    )
    parser.add_argument(
        "--val_file", type=str, 
        default=default_val_file,
        help="Path to validation data JSON."
    )
    parser.add_argument(
        "--test_file", type=str, 
        default=default_test_file,
        help="Path to test data JSON (optional)."
    )
    parser.add_argument(
        "--model_checkpoint", type=str, 
        default=DEFAULT_MODEL_CHECKPOINT, 
        help="Pretrained model checkpoint (e.g., 't5-small', 'google/flan-t5-base')."
    )
    parser.add_argument(
        "--output_dir", type=str, 
        default=default_output_dir,
        help="Directory to save trained model and checkpoints."
    )
    parser.add_argument(
        "--batch_size", type=int, 
        default=BATCH_SIZE, 
        help="Batch size per device."
    )
    parser.add_argument(
        "--epochs", type=int, 
        default=NUM_EPOCHS, 
        help="Number of training epochs."
    )
    parser.add_argument(
        "--learning_rate", type=float, 
        default=LEARNING_RATE, 
        help="Learning rate for fine-tuning."
    )

    args = parser.parse_args()

    # Debug prints to confirm actual paths:
    print("Debug: train_file =", os.path.abspath(args.train_file))
    print("Debug: val_file   =", os.path.abspath(args.val_file))
    print("Debug: test_file  =", os.path.abspath(args.test_file))
    print("Debug: output_dir =", os.path.abspath(args.output_dir))

    # Finally, call your main(args) function (which does the training).
    main(args)
