# Explainable-AI-Based-Credit-Card-Fraud-Detection-Using-Isolation-Forest-and-Autoencoder-Models
Explainable AI-Based Credit Card Fraud Detection Using Isolation Forest and Autoencoder Models
Project Overview

This project implements a credit card fraud detection system using unsupervised machine learning techniques—Isolation Forest and Autoencoder models. The focus is not only on accurate detection but also on making the model interpretable using explainable AI (XAI) techniques such as SHAP.

Fraud detection is critical for financial institutions to prevent unauthorized transactions. This project demonstrates how advanced ML models can be used to identify anomalous transactions and provide interpretability for decision-making.

Features

Detect fraudulent credit card transactions using:

Isolation Forest

Autoencoder Neural Network

Explain model predictions using SHAP (SHapley Additive exPlanations)

Compare global and local feature importances for better interpretability

Visualize anomalies and model performance with plots

Dataset

The project uses the Credit Card Fraud Detection dataset from Kaggle
.

Dataset contains transactions made by European cardholders over two days.

Highly imbalanced with most transactions being legitimate.

Model Evaluation

Isolation Forest: Detects anomalies by isolating outliers in the data.

Autoencoder: Learns normal transaction patterns and flags high reconstruction errors as anomalies.

Performance metrics include precision, recall, F1-score, and accuracy.

SHAP is used to explain which features influenced the prediction for each transaction.

References

Kaggle Credit Card Fraud Detection Dataset: Link

Lundberg, S.M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions.
