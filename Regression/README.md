# Predicting Patient Drug Ratings from Text Reviews

## Project Overview

This project explores the intersection of Natural Language Processing (NLP) and regression modeling to predict patient satisfaction with pharmaceutical drugs. Using the "UCI Drug Review" dataset from Kaggle, this analysis aims to predict a patient's 1-10 star rating based on the text of their review and the associated medical condition.

The core of the project is a comparative analysis of three different modeling architectures to understand the trade-offs between traditional machine learning and modern deep learning approaches for this text-based regression task.

**High-Stakes Context:** While not safety-critical in the same way as wildfire detection, accurately modeling patient sentiment has high stakes for pharmaceutical companies, healthcare providers, and regulatory bodies. It can influence drug development, post-market surveillance, and patient communication strategies.

**Live Demo & Models:**
*   [Link to a live demo if you deploy one using Streamlit/Gradio]
*   [Link to the saved model files if you upload them]

---

## Table of Contents
- [Project Goal](#project-goal)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [1. Data Cleaning and Preparation](#1-data-cleaning-and-preparation)
  - [2. Model Architectures](#2-model-architectures)
- [How to Run This Project](#how-to-run-this-project)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Execution](#execution)
- [Results and Key Findings](#results-and-key-findings)
- [Future Work](#future-work)
- [Technologies Used](#technologies-used)

---

## Project Goal

The primary objective is to build and evaluate a regression model that accurately predicts a patient's drug rating (on a scale of 1-10) from the text of their review. The project systematically compares three models of increasing complexity to determine the most effective approach:
1.  A simple, interpretable baseline (Linear Regression).
2.  A powerful, non-linear traditional model (LightGBM).
3.  A state-of-the-art deep learning model for NLP (1D CNN built with Keras/TensorFlow).

---

## Dataset

This project uses the **[UCI Drug Review Dataset](https://www.kaggle.com/datasets/jessicali9530/kuc-drug-review-dataset)** available on Kaggle.

- **Source:** Data was crawled from online pharmaceutical review sites.
- **Content:** The dataset includes patient reviews on specific drugs, the associated medical condition, a 1-10 star rating, and other metadata.
- **Files:** `drugsComTrain_raw.tsv` and `drugsComTest_raw.tsv`.

A key challenge identified during the initial data loading phase was the presence of missing values in the `condition` column and parsing errors due to inconsistent file formatting. The data preparation pipeline was designed to handle these issues robustly.

---

## Methodology

### 1. Data Cleaning and Preparation

Before modeling, the dataset was rigorously cleaned and prepared:
- **Data Ingestion:** A robust data loading pipeline was created to handle potential `ParserError` issues in the source files.
- **Missing Value Imputation:** The `condition` column contained 899 missing values, which were filled with the placeholder string `"Not Specified"` to avoid data loss.

### 2. Model Architectures

Three models were trained and evaluated:

| Model                 | Vectorization / Embedding         | Algorithm                | Rationale                                                                        |
|-----------------------|-----------------------------------|--------------------------|----------------------------------------------------------------------------------|
| **Linear Regression** | TF-IDF (5000 features, 1-2 ngrams) | Ridge Regression         | A fast, interpretable, and strong baseline to measure against.                   |
| **LightGBM**          | TF-IDF (5000 features, 1-2 ngrams) | Gradient Boosting        | A powerful, non-linear model that excels at tabular/text data.                     |

The deep learning model was specifically chosen as a pivot after initial attempts with RNNs (LSTM/GRU) ran into persistent low-level cuDNN errors in the Colab environment. The 1D CNN architecture provided a powerful, error-free alternative.

---

## How to Run This Project

This project was developed in a Google Colab environment.

### Prerequisites
- Python 3.8+
- A Kaggle account



### Execution

1.  Open the notebook in Google Colab or a local Jupyter environment.
2.  Run all cells in the notebook from top to bottom. The notebook will handle data downloading, cleaning, model training, and evaluation.

---

## Results and Key Findings

The final evaluation yielded insightful results, demonstrating the strengths and weaknesses of each approach.

| Model             | R-squared | MAE    | RMSE   |
|-------------------|-----------|--------|--------|
| Linear Regression | **0.486** | **1.869**| **2.355**|
| LightGBM          | 0.472     | 1.881  | 2.387  |

**Key Findings:**
1.  **Linear Models Prevail:** Surprisingly, the simple Linear Regression model slightly outperformed the more complex LightGBM model, suggesting that the relationship between word frequencies (TF-IDF) and ratings is largely linear.
2.  **The Challenge of Extremes:** All models struggled to predict the extreme ratings (1s and 10s), tending to regress towards the mean. This indicates that the sentiment difference between, for example, a 9-star and a 10-star review is very nuanced and difficult to capture from text alone.


## Future Work

- **Advanced Transformer Models:** Fine-tuning a more powerful transformer model like RoBERTa or T5 could potentially capture more sentiment nuance and improve performance on extreme ratings.
- **Aspect-Based Sentiment Analysis:** Instead of a single rating, break down the problem to predict ratings for specific aspects like "Effectiveness," "Side Effects," and "Ease of Use."
- **Ensemble Modeling:** Combining the predictions of the LightGBM and deep learning models could lead to a more robust and accurate final prediction.

---

## Technologies Used

- **Python 3**
- **Pandas & NumPy** for data manipulation
- **Scikit-learn** for traditional ML models and pipelines (TF-IDF, Ridge, OneHotEncoder)
- **LightGBM** for gradient boosting
- **Hugging Face `datasets`** (initially, before pivoting)
- **Matplotlib & Seaborn** for data visualization
- **Google Colab** for cloud-based GPU computing
