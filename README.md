# Deciphering Financial Documents: A Classification Journey

## Overview
This project aims to classify tables from financial statements into different categories such as Income Statements, Balance Sheets, Cash Flows, Notes, and Others. It utilizes machine learning techniques and natural language processing (NLP) to analyze and classify the text data extracted from HTML files containing financial statement tables.

## Dataset
The dataset consists of HTML files containing tabular data from financial statements. It is organized into five subfolders, each representing a different document category.

## Project Structure
- `data`: Contains the dataset of HTML files organized by document category.
- `models`: Stores the pre-trained machine learning models for text classification.
- `src`: Contains the source code for data preprocessing, model training, and evaluation.
  - `preprocess.py`: Includes functions for cleaning text data and extracting features from HTML files.
  - `train.py`: Script for training the classification model.
  - `evaluate.py`: Script for evaluating the trained model's performance.
- `requirements.txt`: Lists all the Python dependencies required to run the project.

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/financial-document-classification.git
