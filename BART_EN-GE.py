# ================================================================================
# BART Machine Translation: English to German (WMT14)
# ================================================================================
# This script fine-tunes Facebook's BART model for neural machine translation (NMT).
# It trains BART to translate English sentences into German using the WMT14 dataset.
#
# Task: Machine translation (seq2seq)
# Model: BART (Bidirectional Auto-Regressive Transformer)
# Dataset: WMT14 English-German (de-en) from HuggingFace
# Goal: Train BART to learn to translate English text to German
#
# Key Steps:
#   1. Load the WMT14 English-German dataset from HuggingFace
#   2. Preprocess: English as input, German as target translation
#   3. Tokenize both source and target languages
#   4. Fine-tune BART with BLEU metric evaluation
#   5. Generate German translations for new English sentences
#
# Metrics tracked: Loss, BLEU score (standard metric for machine translation)
# BLEU measures n-gram overlap between generated and reference translations
# ================================================================================

# 1) Import libraries
from datasets import load_dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)
import numpy as np
import evaluate

# 2) Load WMT14 English–German dataset
wmt = load_dataset("wmt14", "de-en")

# Inspect one example
print(wmt["train"][0])
# {'translation': {'en': 'Hello world', 'de': 'Hallo Welt'}}

# 3) Load tokenizer and model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# 4) Preprocessing function
max_input_len = 128
max_target_len = 128

def preprocess(example):
    # English input
    inputs = tokenizer(
        example["translation"]["en"],
        max_length=max_input_len,
        truncation=True,
        padding="max_length"
    )
    # German target
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["translation"]["de"],
            max_length=max_target_len,
            truncation=True,
            padding="max_length"
        )

    # Replace pad tokens with -100 so loss ignores them
    label_ids = [id if id != tokenizer.pad_token_id else -100 for id in labels["input_ids"]]
    inputs["labels"] = label_ids
    return inputs

# 5) Apply preprocessing to train/validation sets
train_data = wmt["train"].map(preprocess, remove_columns=wmt["train"].column_names)
val_data   = wmt["validation"].map(preprocess, remove_columns=wmt["validation"].column_names)

# 6) Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# 7) Evaluation metric (BLEU)
bleu = evaluate.load("bleu")
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute BLEU
    result = bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
    return {"bleu": result["bleu"]}

# 8) Training arguments
training_args = TrainingArguments(
    output_dir="./bart-wmt14",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    predict_with_generate=True,
    logging_steps=100
)

# 9) Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 10) Train
trainer.train()

# 11) Evaluate
eval_results = trainer.evaluate()
print("Evaluation:", eval_results)

# 12) Inference (translate new sentence)
input_text = "How are you today?"
inputs = tokenizer([input_text], return_tensors="pt")

generated_ids = model.generate(
    inputs["input_ids"],
    max_length=50,
    num_beams=5,
    length_penalty=1.0,
    no_repeat_ngram_size=3,
    early_stopping=True
)

print("Translation:", tokenizer.decode(generated_ids[0], skip_special_tokens=True))
