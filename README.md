# Phishing Email Detection using Scikit-learn

## Overview
This project builds a machine learning model to classify emails as either **Phishing** or **Safe** using Scikit-learn.

## Features
- Train on phishing and legitimate emails
- Extract text features using TF-IDF
- Classify emails as Phishing or Safe
- Display accuracy and confusion matrix

## Dataset
Dataset: kaggle Phishing_Email.csv
Dataset link : https://www.kaggle.com/datasets/subhajournal/phishingemails?
## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib

## Machine Learning Model
- TF-IDF Vectorization
- Multinomial Naive Bayes

## Results
- Accuracy: 94.58%
- Precision: 95%
- Recall: 94%
- F1-Score: 95%

## How to Run

```bash
pip install -r requirements.txt
python phishing.py
