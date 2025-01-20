# Deep Learning Methods for Clinical Workflow Phase-Based Prediction of Procedure Duration
# Overview
This repository contains the code and resources for the paper "Deep Learning Methods for Clinical Workflow Phase-Based Prediction of Procedure Duration: A Benchmark Study."
Our study explores the performance of deep learning models in predicting procedure end times in cardiac catheterization laboratories, using clinical phases derived from video analysis as input.

Key findings highlight the effectiveness of CNN-based architectures, particularly InceptionTime, which achieves high accuracy with Mean Absolute Error (MAE) below 5 minutes and Symmetric Mean Absolute Percentage Error (SMAPE) under 6%.
## Built with
<img src="./pycache/TensorFlow-Dark.svg" width="48">  <img src="./pycache/Scikit_learn_logo_small.svg" width="48"> <img src="./pycache/NumPy_logo_2020.svg" width="48"> <img src="./pycache/Keras_logo.svg" width="48">  

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

# Getting Started
## Prerequisites
To install the required dependencies, run this command:
  ```sh
  pip install -r requirements.txt
  ```
## Dataset
The dataset used in this study is stored in a folder named `data`, located at the root of the repository. It consists of two CSV files:
 * `X.csv`: Contains the input features, namely clinical phases derived from video analysis.
 * `Y.csv`: Contains the target labels, representing the time to the end of the procedure, in seconds.
 
Both input and output are recorded with the frequency of 1 datapoint per second.
Ensure the dataset files are placed in the `data` folder before running the code.


# Contact
Emanuele Frassini - ema.frassini@hotmail.com -[Emanuele Frassini](https://www.linkedin.com/in/emanuele-frassini-1a7a37208)