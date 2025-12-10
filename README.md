# BART WMT14 English to German Translation

This repository contains a full implementation for fine-tuning **Facebook’s BART (Bidirectional and Auto-Regressive Transformer)** model for **Neural Machine Translation (NMT)** using the **WMT14 English–German (de-en)** dataset.

The model is trained to translate:
English → German

---

## 🚀 Project Overview

- **Task**: Machine Translation (Seq2Seq)
- **Model**: `facebook/bart-base`
- **Dataset**: WMT14 English–German
- **Metric**: BLEU Score
- **Frameworks**: Hugging Face Transformers, Datasets, Evaluate
- **Training Method**: Supervised fine-tuning

---

## 🧠 What the Model Learns

- Sentence-level bilingual translation
- Proper syntactic and semantic mapping between English and German
- Word reordering and contextual transfer learning

---

## 📦 Installation

pip install transformers datasets evaluate torch accelerate

## 📁 Dataset Loading

The dataset is loaded directly from Hugging Face:

from datasets import load_dataset
wmt = load_dataset("wmt14", "de-en")

## 📊 Evaluation (BLEU Metric)

BLEU measures the n-gram overlap between the model's translation and the reference translation.

BLEU Score = Translation Quality Indicator

## ✨ Inference

input_text = "How are you today?"

## ✅ Model Output:

Wie geht es dir heute?

## 📊 Key Features

Proper label masking using -100

Dynamic padding with DataCollatorForSeq2Seq

BLEU score-based evaluation

Beam search for high-quality translations

Automatic tokenization of bilingual text

Clean Hugging Face training loop

## 🧪 Use Cases

Multilingual chatbots

Language translation tools

NLP research projects

International content processing

AI-powered localization pipelines

## 📌 Future Improvements

Add SacreBLEU & ChrF metrics

Add low-resource training mode

Web-based translation demo

Export trained model to Hugging Face Hub

Mixed precision (FP16) training

## 📜 License

This project is released under the MIT License.

Feel free to modify, experiment, and improve this project!
