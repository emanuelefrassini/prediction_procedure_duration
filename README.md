# Deep Learning Methods for Clinical Workflow Phase-Based Prediction of Procedure Duration
# Overview
This repository contains the code and resources for the paper "Deep Learning Methods for Clinical Workflow Phase-Based Prediction of Procedure Duration: A Benchmark Study."
Our study explores the performance of deep learning models in predicting procedure end times in cardiac catheterization laboratories (cath labs), using clinical phases derived from video analysis as input.

Key findings highlight the effectiveness of CNN-based architectures, particularly InceptionTime, which achieves high accuracy with Mean Absolute Error (MAE) below 5 minutes and Symmetric Mean Absolute Percentage Error (SMAPE) under 6%.

# Key Features
Implements state-of-the-art deep learning models, including:
 * InceptionTime
 * Transformer
 * LSTM-FCN
 * LSTM
 * LSTM with Attention layer
 * Ensemble

Benchmarks models based on:
 * Accuracy (MAE, SMAPE)
 * Training and inference times
