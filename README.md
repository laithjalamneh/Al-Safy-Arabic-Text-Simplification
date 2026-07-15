# Al-Safy | الصافي

Arabic Text Simplification System using OCR, AraBART, and Custom Transformer Training.

---

## Overview

Al-Safy is an Arabic NLP system designed to simplify complex Arabic legal and formal text while preserving the original meaning.

The system combines OCR technology with a fine-tuned AraBART transformer model to make official Arabic documents more accessible to non-experts, students, and the general public.

---

## Problem Statement

Arabic legal and administrative documents often contain:

- Complex sentence structures
- Formal terminology
- Long clauses
- Difficult vocabulary

These characteristics make documents difficult to understand for many Arabic speakers.

Al-Safy aims to bridge this gap by transforming complex text into simpler Arabic while preserving legal meaning.

---

## System Architecture

Image Upload
↓
OCR Extraction
↓
AraBART Simplification Model
↓
Post Processing Pipeline
↓
Simplified Arabic Output

---

## Features

- Arabic OCR integration
- Arabic legal text simplification
- Transformer-based sequence-to-sequence model
- Custom training pipeline
- Post-processing and quality filtering
- Web-based interface

---

## Dataset

The model was trained using:

### Jordanian Legal Simplification Dataset

- 2,000 legal text pairs

### Filtered News Simplification Dataset

- 2,307 pairs

### Data Augmentation

Synonym augmentation expanded the dataset from:

4,307 pairs → 9,185 pairs

Dataset Split:

- Train: 7,348
- Validation: 918
- Test: 919

---

## Model

Base Model:

AraBART

Training Hardware:

NVIDIA RTX 3060 Ti (8GB)

Training Configuration:

- Batch Size: 1
- Gradient Accumulation: 16
- Optimizer: Adafactor
- Learning Rate: 2e-5
- Mixed Precision: bf16
- Early Stopping
- Cosine Learning Rate Scheduler

---

## Custom Loss Components

The model extends standard Cross Entropy Loss using:

### Label Smoothing

Reduces model overconfidence and repetitive outputs.

### Length Penalty

Encourages explanatory outputs rather than summarization.

### Coverage Loss

Ensures the model attends to the entire input text.

### Grammar Penalty

Penalizes grammatically broken Arabic outputs.

---

## Post-Processing Pipeline

The output passes through a 7-step cleaning pipeline:

1. Remove leading connectors
2. Remove boilerplate phrases
3. Fix grammar artifacts
4. Remove garbage suffixes
5. Remove invalid lines
6. Remove duplicate sentences
7. Limit output length

This significantly improves output quality.

---

## Evaluation Results

ROUGE Scores:

| Metric | Score |
|----------|----------|
| ROUGE-1 | 51.67% |
| ROUGE-2 | 46.26% |
| ROUGE-L | 47.31% |

Fallback Rate:

1.4%

---

## Technology Stack

### NLP

- AraBART
- HuggingFace Transformers
- PyTorch
- Datasets

### Backend

- Flask

### Frontend

- React

### OCR

- Mistral OCR

### Data Processing

- Pandas
- NumPy

---

## Project Structure

```text
Al-Safy-Arabic-Text-Simplification
│
├── train.py
├── requirements.txt
├── README.md
│
├── screenshots/
│
├── frontend/
│
└── backend/
```

## Future Work

- Larger human-verified Arabic simplification datasets
- Better domain generalization
- Mobile application support
- Real-time document simplification

---

## Authors

Laith Yousef

Ahmad Atiyat

Mohammad Alrashdan

Al Hussein Technical University (HTU)

---

## License

MIT License
