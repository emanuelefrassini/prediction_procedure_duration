# -*- coding: utf-8 -*-
"""
Created on Tue May 14 11:14:54 2024

@author: emanuelefrassi
"""

import os
import numpy as np
#import pandas as pd
from sklearn.metrics import mean_absolute_error
def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum( np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# Define the path to the main folder
main_folder = r'C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Codes\MTSC\runs\runs_PAPER_20240429-121956'

#PARAMETERS
REDUCE_DIM=60

perc_INCEPTIONTIME=0.5
perc_LSTM_FCN=0.5
perc_transformer=0
perc_LSTM=0
perc_LSTM_Attention=0

##################### 5 seconds ################################################
# Initialize lists to store accuracy numbers for each trial
# Loop through each trial folder
length=len( os.listdir(main_folder))-1

ground_truth_INCEPTIONTIME_5=np.full((length,3,74,2000),-1)
prediction_array_INCEPTIONTIME_5=np.full((length,3,74,2000),-1)

ground_truth_LSTM_5=np.full((length,3,74,2000),-1)
prediction_array_LSTM_5=np.full((length,3,74,2000),-1)

ground_truth_LSTM_FCN_5=np.full((length,3,74,2000),-1)
prediction_array_LSTM_FCN_5=np.full((length,3,74,2000),-1)

ground_truth_LSTM_Attention_5=np.full((length,3,74,2000),-1)
prediction_array_LSTM_Attention_5=np.full((length,3,74,2000),-1)

ground_truth_transformer_5=np.full((length,3,74,2000),-1)
prediction_array_transformer_5=np.full((length,3,74,2000),-1)


i=0
for trial_folder in os.listdir(main_folder):   
    
    if trial_folder.startswith('trial_') and trial_folder.endswith('.xlsx')==False:# and int(trial_folder[-2:])!=(len(os.listdir(main_folder))-1):

        trial_path = os.path.join(main_folder, f'{trial_folder}')
        # Loop through each fold folder
        reduce_dim_path_5=os.path.join(trial_path, f'reduce_dim_{REDUCE_DIM}')
        j=0
        for fold_folder in os.listdir(reduce_dim_path_5):
        
            if fold_folder.startswith('fold_'):
                fold_path_5=os.path.join(reduce_dim_path_5, fold_folder)
                
                ############   INCEPTIONTIME  ########################
                INCEPTIONTIME_path_5 = os.path.join(fold_path_5, 'INCEPTIONTIME')
                prediction_path_5=os.path.join(INCEPTIONTIME_path_5,'test_predictions')

                for number in range(74):                                   
                    ground_truth_5 = []
                    prediction_5 = []
                    # Open the text file and read line by line
                    with open(os.path.join(prediction_path_5, f'test_predictions_{number}.txt'), 'r') as file:
                        for line in file:
                            # Find the positions of the square brackets for truth
                            truth_start = line.find('Truth: [') + len('Truth: [')
                            truth_end = line.find(']', truth_start)
                            truth_value = float(line[truth_start:truth_end])
                    
                            # Find the positions of the square brackets for prediction
                            pred_start = line.find('prediction: [') + len('prediction: [')
                            pred_end = line.find(']', pred_start)
                            pred_value = float(line[pred_start:pred_end])
                    
                            # Append the extracted values to the respective lists
                            ground_truth_5.append(truth_value)
                            prediction_5.append(pred_value)
                    ground_truth_INCEPTIONTIME_5[i,j,number,0:len(ground_truth_5)]=ground_truth_5
                    prediction_array_INCEPTIONTIME_5[i,j,number,0:len(prediction_5)]=prediction_5

                ############   LSTM  ########################
                LSTM_path_5 = os.path.join(fold_path_5, 'LSTM')
                prediction_path_5=os.path.join(LSTM_path_5,'test_predictions')

                for number in range(74):                                   
                    ground_truth_5 = []
                    prediction_5 = []
                    # Open the text file and read line by line
                    with open(os.path.join(prediction_path_5, f'test_predictions_{number}.txt'), 'r') as file:
                        for line in file:
                            # Find the positions of the square brackets for truth
                            truth_start = line.find('Truth: [') + len('Truth: [')
                            truth_end = line.find(']', truth_start)
                            truth_value = float(line[truth_start:truth_end])
                    
                            # Find the positions of the square brackets for prediction
                            pred_start = line.find('prediction: [') + len('prediction: [')
                            pred_end = line.find(']', pred_start)
                            pred_value = float(line[pred_start:pred_end])
                    
                            # Append the extracted values to the respective lists
                            ground_truth_5.append(truth_value)
                            prediction_5.append(pred_value)
                    ground_truth_LSTM_5[i,j,number,0:len(ground_truth_5)]=ground_truth_5
                    prediction_array_LSTM_5[i,j,number,0:len(prediction_5)]=prediction_5
                    
                ############   LSTM_FCN  ########################
                LSTM_FCN_path_5 = os.path.join(fold_path_5, 'LSTM_FCN')
                prediction_path_5=os.path.join(LSTM_FCN_path_5,'test_predictions')

                for number in range(74):                                   
                    ground_truth_5 = []
                    prediction_5 = []
                    # Open the text file and read line by line
                    with open(os.path.join(prediction_path_5, f'test_predictions_{number}.txt'), 'r') as file:
                        for line in file:
                            # Find the positions of the square brackets for truth
                            truth_start = line.find('Truth: [') + len('Truth: [')
                            truth_end = line.find(']', truth_start)
                            truth_value = float(line[truth_start:truth_end])
                    
                            # Find the positions of the square brackets for prediction
                            pred_start = line.find('prediction: [') + len('prediction: [')
                            pred_end = line.find(']', pred_start)
                            pred_value = float(line[pred_start:pred_end])
                    
                            # Append the extracted values to the respective lists
                            ground_truth_5.append(truth_value)
                            prediction_5.append(pred_value)
                    ground_truth_LSTM_FCN_5[i,j,number,0:len(ground_truth_5)]=ground_truth_5
                    prediction_array_LSTM_FCN_5[i,j,number,0:len(prediction_5)]=prediction_5

                ############   LSTM_Attention  ########################
                LSTM_Attention_path_5 = os.path.join(fold_path_5, 'LSTM_Attention')
                prediction_path_5=os.path.join(LSTM_Attention_path_5,'test_predictions')

                for number in range(74):                                   
                    ground_truth_5 = []
                    prediction_5 = []
                    # Open the text file and read line by line
                    with open(os.path.join(prediction_path_5, f'test_predictions_{number}.txt'), 'r') as file:
                        for line in file:
                            # Find the positions of the square brackets for truth
                            truth_start = line.find('Truth: [') + len('Truth: [')
                            truth_end = line.find(']', truth_start)
                            truth_value = float(line[truth_start:truth_end])
                    
                            # Find the positions of the square brackets for prediction
                            pred_start = line.find('prediction: [') + len('prediction: [')
                            pred_end = line.find(']', pred_start)
                            pred_value = float(line[pred_start:pred_end])
                    
                            # Append the extracted values to the respective lists
                            ground_truth_5.append(truth_value)
                            prediction_5.append(pred_value)
                    ground_truth_LSTM_Attention_5[i,j,number,0:len(ground_truth_5)]=ground_truth_5
                    prediction_array_LSTM_Attention_5[i,j,number,0:len(prediction_5)]=prediction_5

                ############   transformer  ########################
                transformer_path_5 = os.path.join(fold_path_5, 'transformer')
                prediction_path_5=os.path.join(transformer_path_5,'test_predictions')

                for number in range(74):                                   
                    ground_truth_5 = []
                    prediction_5 = []
                    # Open the text file and read line by line
                    with open(os.path.join(prediction_path_5, f'test_predictions_{number}.txt'), 'r') as file:
                        for line in file:
                            # Find the positions of the square brackets for truth
                            truth_start = line.find('Truth: [') + len('Truth: [')
                            truth_end = line.find(']', truth_start)
                            truth_value = float(line[truth_start:truth_end])
                    
                            # Find the positions of the square brackets for prediction
                            pred_start = line.find('prediction: [') + len('prediction: [')
                            pred_end = line.find(']', pred_start)
                            pred_value = float(line[pred_start:pred_end])
                    
                            # Append the extracted values to the respective lists
                            ground_truth_5.append(truth_value)
                            prediction_5.append(pred_value)
                    ground_truth_transformer_5[i,j,number,0:len(ground_truth_5)]=np.array(ground_truth_5)
                    prediction_array_transformer_5[i,j,number,0:len(prediction_5)]=np.array(prediction_5)
                j+=1
        i+=1   
#%%
ground_truth_ensemble_5=ground_truth_transformer_5      
prediction_ensemble_5=perc_transformer*prediction_array_transformer_5+perc_INCEPTIONTIME*prediction_array_INCEPTIONTIME_5+perc_LSTM_FCN*prediction_array_LSTM_FCN_5+perc_LSTM*prediction_array_LSTM_5        

prediction_ensemble_5=prediction_array_LSTM_FCN_5

mae_trial_5=[]
smape_trial_5=[]

std_mae_5=[]
std_smape_5=[]

for trialnum in range(len(os.listdir(main_folder))-2):
    mae_5=[]
    smape_5=[]
    for foldnum in range(3):
        for h in range(74):
            indices = np.where(ground_truth_ensemble_5[trialnum,foldnum,h] == -1)[0][0]
           # print(smape(ground_truth_ensemble_5[trialnum,foldnum,0], prediction_ensemble_5[trialnum,foldnum,0]))
            smape_5.append(smape(ground_truth_ensemble_5[trialnum,foldnum,h,0:indices], prediction_ensemble_5[trialnum,foldnum,h,0:indices]))
            #smape_5.append(smape(ground_truth_ensemble_5[39,foldnum,h], prediction_ensemble_5[39,foldnum,h]))

            mae_5.append(mean_absolute_error(ground_truth_ensemble_5[trialnum,foldnum,h,0:indices], prediction_ensemble_5[trialnum,foldnum,h,0:indices]))
    mae_trial_5.append(np.nanmean(mae_5))
    smape_trial_5.append(np.nanmean(smape_5))
    
    std_mae_5.append(np.std(mae_5))
    std_smape_5.append(np.std(smape_5))
mae_ensemble_5=np.nanmean(mae_trial_5)
smape_ensemble_5=np.nanmean(smape_trial_5)
print('MAE: ', mae_ensemble_5/60)
print('std MAE \n', np.std(mae_trial_5)/60)
print('SMAPE: ', smape_ensemble_5)
print('std SMAPE', np.std(smape_trial_5))

#%%
ground_truth_ensemble_5=ground_truth_transformer_5      
prediction_ensemble_5=perc_transformer*prediction_array_transformer_5+perc_INCEPTIONTIME*prediction_array_INCEPTIONTIME_5+perc_LSTM_FCN*prediction_array_LSTM_FCN_5+perc_LSTM*prediction_array_LSTM_5+perc_LSTM_Attention*prediction_array_LSTM_Attention_5    
prediction_ensemble_5=prediction_array_transformer_5
def find_closest_index(arr, target=100):
    closest_index = None
    closest_diff = float('inf')
    
    for index, value in enumerate(arr):
        diff = abs(value - target)
        if diff < closest_diff:
            closest_diff = diff
            closest_index = index
            
    return closest_index

BESTELD_TIME=12.33
BESTELD_TIME=BESTELD_TIME*60
error_per_trial=[]

for i in range(prediction_ensemble_5.shape[0]):
    error_per_fold=[]
    for j in range(prediction_ensemble_5.shape[1]):
        error=[]
        for h in range(prediction_ensemble_5.shape[2]):
            index=find_closest_index(ground_truth_transformer_5[i,j,h],BESTELD_TIME)
            pred=prediction_ensemble_5[i,j,h,index]
            error.append(np.abs(prediction_ensemble_5[i,j,h,index]-ground_truth_transformer_5[i,j,h,index]))
        error_per_fold.append(np.mean(error))
    error_per_trial.append(np.mean(error_per_fold))
error_total=np.mean(error_per_trial)
print(f"When {BESTELD_TIME/60} minutes are missing the average error is of {error_total} seconds")
            


 #%%   
# Plot the data
data.plot(kind='bar', figsize=(10, 6))
plt.title('Error at different time points')
plt.xlabel('Model')
plt.ylabel('Time [seconds]')
plt.xticks(ticks=range(len(df.index)), labels=df.Model,rotation=45)
plt.rcParams.update({
    'font.size': 14,         # General font size
    'axes.titlesize': 16,    # Title font size
    'axes.labelsize': 16,    # X and Y label font size
    'xtick.labelsize': 12,   # X-axis tick label font size
    'ytick.labelsize': 14,   # Y-axis tick label font size
    'legend.fontsize': 12,   # Legend font size
    'figure.titlesize': 18   # Figure title font size
})
plt.legend()
plt.show()
