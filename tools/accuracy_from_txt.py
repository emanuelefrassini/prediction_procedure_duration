# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:31:49 2024

@author: emanuelefrassi
"""

import os
import re
import numpy as np
from sklearn.metrics import mean_absolute_error
# Define the folder path where the text files are located
folder_path = r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Codes\MTSC\runs\runs_PAPER_20240426-164502\reduce_dim_60\fold_2\LSTM_FCN\test_predictions"

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

treshold=0

def compute_accuracy(arr):
    count = 0
    length=arr.shape[0]
    for element in arr:
        if element == 1:
            count += 1
    return count/length

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

y_test_copy=y_test.copy()
y_test_copy_predict=y_test_predict.copy()
REDUCE_DIM=5

mae=mean_absolute_error(y_test,y_test_predict)/60
indices1=[]
mae2list=[]
for i in range(y_test.shape[0]):
    indices1.append(int(np.where(y_test[i]==-1)[0][0]))
    mae2=mean_absolute_error(y_test[i,0:indices1[i]], y_test_predict[i,0:indices1[i]])
    mae2list.append(mae2)
print(np.mean(mae2)/60)
print(mae)
mae3=[]
mae4=[]
indices2=[]
for i in range(y_test.shape[0]):

    indices = np.where(y_test[i] == -1)[0]
    if len(indices) == 0:
        indices=int(np.ceil(7573/REDUCE_DIM))
    else:
        indices=indices[0]
   
    indices2.append(indices)
    mae3.append(mean_absolute_error(y_test_rescaled[i,0:indices], y_test_predict_LSTM_FCN[i,0:indices]))
    mae3.append(mean_absolute_error(y_test[i,0:indices], y_test_predict[i,0:indices]))
    mae4.append(mean_absolute_error(y_test[i], y_test_predict[i]))


print(np.mean(mae3)/60)
print(np.mean(mae4)/60)
mae4=[item/60 for item in mae4]
print(np.mean(mae2list)/60)
print(mae)



correct_5=np.zeros((y_test_copy.shape[0],1))
error_5=np.zeros((y_test_copy.shape[0],1))
correct_10=np.zeros((y_test_copy.shape[0],1))
error_10=np.zeros((y_test_copy.shape[0],1))
correct_15=np.zeros((y_test_copy.shape[0],1))
error_15=np.zeros((y_test_copy.shape[0],1))
correct_20=np.zeros((y_test_copy.shape[0],1))
error_20=np.zeros((y_test_copy.shape[0],1))

for i in range(y_test_copy.shape[0]):
    ind_5 = (np.abs(y_test_copy_predict[i] -60*5)).argmin()
    error_5[i]=np.abs(y_test_copy[i,ind_5]-60*5)
    if y_test_copy[i,ind_5]<60*5 and y_test_copy[i,ind_5]!=-1:
        correct_5[i]=1
        
    ind_10 = (np.abs(y_test_copy_predict[i] -60*10)).argmin()
    error_10[i]=np.abs(y_test_copy[i,ind_10]-60*10)
    if y_test_copy[i,ind_10]<60*10 and y_test_copy[i,ind_10]!=-1:
          correct_10[i]=1

    ind_15 = (np.abs(y_test_copy_predict[i] -60*15)).argmin()
    error_15[i]=np.abs(y_test_copy[i,ind_15]-60*15)
    if y_test_copy[i,ind_15]<60*15 and y_test_copy[i,ind_15]!=-1:
           correct_15[i]=1

    ind_20 = (np.abs(y_test_copy_predict[i] -60*20)).argmin()
    error_20[i]=np.abs(y_test_copy[i,ind_20]-60*20)

    #  ind=np.where( abs(y_test_copy_predict[i]-60*howmany)<REDUCE_DIM)[0][0]
    if y_test_copy[i,ind_20]<60*20 and y_test_copy[i,ind_20]!=-1:
          correct_20[i]=1



for i in range(error_5.shape[0]):
    if error_5[i]<= treshold:
        correct_5[i]=1
    if error_10[i]<= treshold:
        correct_10[i]=1
    if error_15[i]<= treshold:
        correct_15[i]=1
    if error_20[i]<= treshold:
        correct_20[i]=1



accuracy_5=compute_accuracy(correct_5)
accuracy_10=compute_accuracy(correct_10)
accuracy_15=compute_accuracy(correct_15)

accuracy_20=compute_accuracy(correct_20)


print(f"\nAccuracy at 5: {accuracy_5} \nAccuracy at 10: {accuracy_10} \nAccuracy at 15: {accuracy_15} \nAccuracy at 20: {accuracy_20}")         


