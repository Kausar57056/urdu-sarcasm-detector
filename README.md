# Urdu Sarcasm Detector 🤖

A deep learning-based web application that detects sarcasm in Urdu tweets using a custom hybrid model named SentimixtureNet.

## Overview:

This project addresses the challenge of sarcasm detection in Urdu tweets—a task complicated by cultural context, linguistic subtleties, and the lack of labeled datasets. The system is trained using a custom neural architecture that blends transformer features with routing and expert attention layers.

### Achievement:
- Achieved 0.8083 validation accuracy (XLM-RoBERTa + SentimixtureNet)
- Custom preprocessing pipeline for Urdu text
- Web app built with Streamlit for real-time tweet sarcasm detection

## Model: SentimixtureNet

- Base: xlm-roberta-base
- Routing Network (Linear + ReLU)
- Expert Attention (Multi-head Attention)
- Final Classifier Layer

## Project Structure:

```bash
urdu-sarcasm-detector/
├── app.py                    # Streamlit frontend
├── train.py                 # Model training script
├── preprocess.py           # Data cleaning and prep
├── urdu_sarcastic_dataset.csv
├── requirements.txt
└── best_model/             # Saved model

## Dataset:

The dataset urdu_sarcastic_dataset.csv contains:
	•	urdu_text: the tweet text
	•	is_sarcastic: label (0 for not sarcastic, 1 for sarcastic)

## How to Run?

Install requirements
Train Model (optional)
streamlit run app.py

## Future Work:

Support more low-resource regional languages
Live tweet scrapping from X
Improve model accuracy with larger transformer variants

## License:

MIT License
