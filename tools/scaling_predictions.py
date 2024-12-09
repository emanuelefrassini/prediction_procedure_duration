# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:21:12 2024

@author: emanuelefrassi
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import os
import re

# Define the folder path where the text files are located
folder_path = r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Results\Prediction of duration\best_runs\reduce_dim_5\runs_20240408-100513_time_binary_regression\test_predictions"

# Initialize lists to store values
y_test_list = []
y_test_predict_list = []
y_test=np.full((len(os.listdir(folder_path)),5000), -1)
y_test_predict=np.full((len(os.listdir(folder_path)),5000), -1)

# Define a regular expression pattern to extract the values
#pattern = r"Truth: \[(\d+\.\d+)\], prediction: \[(\d+\.\d+)\]"
pattern = r"Truth: \[(.*?)\], prediction: \[(.*?)\]"

i=0
massimo=50

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
        y_test_list = []
        y_test_predict_list = []
        file_path = os.path.join(folder_path, filename)
        
        # Open and read the contents of the file
        with open(file_path, 'r') as file:
            contents = file.read()
        
        # Use regex to find all matches of the pattern in the file contents
        matches = re.findall(pattern, contents)
        
        # Extract values from each match and append to the lists
        for match in matches:
            truth_value = float(match[0])  # Extract value after "Truth: ["
            predict_value = float(match[1])  # Extract value after "prediction: ["
            y_test_list.append(truth_value)
            y_test_predict_list.append(predict_value)
        y_test[i,:len(y_test_list)]=y_test_list
        y_test_predict[i,:len(y_test_list)]=y_test_predict_list
        if massimo<len(y_test_list):
            massimo=len(y_test_list)
        i+=1
y_test=y_test[:,:massimo]
y_test_predict=y_test_predict[:,:massimo]






scaler = MinMaxScaler()

# Fit and transform y_true
y_true_scaled = scaler.fit_transform(y_test)

# Transform y_pred using the same scaler
y_pred_scaled = scaler.transform(y_test_predict)
    
min_val = np.min(y_test)
max_val = np.max(y_test)

# Scale y_true to [0, 1]
y_true_scaled = (y_test - min_val) / (max_val - min_val)
y_pred_scaled = (y_test_predict - min_val) / (max_val - min_val)

y_true_average=np.mean(y_true_scaled,axis=0)
y_pred_average=np.mean(y_pred_scaled,axis=0)

pos=np.where(y_true_average==0)[0][0]

plt.figure(figsize=(10, 6))  # Set the figure size

# Plot y_true_scaled (blue line) and y_pred_scaled (orange line)
plt.plot(y_true_average[:pos], label='Real ETC', color='blue', linestyle='--')
plt.plot(y_pred_average[:pos], label='Predicted ETC', color='orange', linestyle='-')
# Adding labels and title
plt.xlabel('Time [min]')
plt.ylabel('Scaled ETC')
plt.title('Scaled average among all the procedures of real and predicted ETC')
plt.legend()  # Show legend

x_ticks_positions = [0,60/5*10,60/5*20,60/5*30,60/5*40,60/5*50,60/5*60,60/5*70,60/5*80,60/5*90]
#[200,400,600,800,1000,1200,1400]
#np.arange(len(y_true_average)) * 5
x_ticks_labels = [int(num*5/60) for num in x_ticks_positions]


# Set the x ticks using the defined positions and labels
plt.xticks(x_ticks_positions, x_ticks_labels)

# Display the plot
plt.grid(True)  # Add grid
plt.show()