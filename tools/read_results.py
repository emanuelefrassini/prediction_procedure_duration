# -*- coding: utf-8 -*-
"""
Created on Tue May 14 11:14:54 2024

@author: emanuelefrassi
"""

import os
import numpy as np
import pandas as pd
#import re

# Define the path to the main folder
main_folder = r'C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Codes\MTSC\runs\runs_PAPER_20240429-121956'



##################### 5 seconds ################################################
# Initialize lists to store accuracy numbers for each trial
average_mae_INCEPTIONTIME_5 = []
average_SMAPE_INCEPTIONTIME_5= []
average_traintime_INCEPTIONTIME_5=[]
average_testtime_INCEPTIONTIME_5=[]


average_mae_LSTM_5 = []
average_SMAPE_LSTM_5= []
average_traintime_LSTM_5=[]
average_testtime_LSTM_5=[]

average_mae_LSTM_FCN_5 = []
average_SMAPE_LSTM_FCN_5= []
average_traintime_LSTM_FCN_5=[]
average_testtime_LSTM_FCN_5=[]

average_mae_transformer_5 = []
average_SMAPE_transformer_5= []
average_traintime_transformer_5=[]
average_testtime_transformer_5=[]

average_mae_LSTM_Attention_5 = []
average_SMAPE_LSTM_Attention_5= []
average_traintime_LSTM_Attention_5=[]
average_testtime_LSTM_Attention_5=[]

average_mae_Ensemble_5 = []
average_SMAPE_Ensemble_5= []
# Loop through each trial folder
for trial_folder in os.listdir(main_folder):    
    if trial_folder.startswith('trial_') and trial_folder.endswith('.xlsx')==False:# and trial_folder.endswith(str(len(os.listdir(main_folder))-2))==False and int(trial_folder[-2:])!=(len(os.listdir(main_folder))-1):

        trial_path = os.path.join(main_folder, f'{trial_folder}')
        # Loop through each fold folder
        reduce_dim_path_5=os.path.join(trial_path, 'reduce_dim_5')
        
        mae_INCEPTIONTIME_5 = []
        SMAPE_INCEPTIONTIME_5= []
        traintime_INCEPTIONTIME_5=[]
        testtime_INCEPTIONTIME_5=[]
        
        mae_LSTM_5 = []
        SMAPE_LSTM_5= []
        traintime_LSTM_5=[]
        testtime_LSTM_5=[]
        
        mae_LSTM_FCN_5 = []
        SMAPE_LSTM_FCN_5= []
        traintime_LSTM_FCN_5=[]
        testtime_LSTM_FCN_5=[]
        
        mae_transformer_5 = []
        SMAPE_transformer_5= []
        traintime_transformer_5=[]
        testtime_transformer_5=[]
        
        mae_LSTM_Attention_5 = []
        SMAPE_LSTM_Attention_5= []
        traintime_LSTM_Attention_5=[]
        testtime_LSTM_Attention_5=[]
        
        mae_Ensemble_5 = []
        SMAPE_Ensemble_5= []
        
        for fold_folder in os.listdir(reduce_dim_path_5):
            if fold_folder.startswith('fold_'):
                fold_path_5=os.path.join(reduce_dim_path_5, fold_folder)
                
                ############   INCEPTION  ########################
                INCEPTIONTIME_path_5 = os.path.join(fold_path_5, 'INCEPTIONTIME')
         #       print((os.path.join(INCEPTIONTIME_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(INCEPTIONTIME_path_5, 'mae_per_procedure.txt'), 'r') as file:
                    mae_INCEPTIONTIME_5.append(float(file.readline()[48:54]))
                with open(os.path.join(INCEPTIONTIME_path_5, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_INCEPTIONTIME_5.append(float(file.readline()[44:50]))
                with open(os.path.join(INCEPTIONTIME_path_5, 'time.txt'), 'r') as file:
                    traintime_INCEPTIONTIME_5.append(float(file.readline()[17:23]))
                    testtime_INCEPTIONTIME_5.append(float(file.readline()[16:22]))
                                        
                
                ############   LSTM  ########################
                LSTM_path_5 = os.path.join(fold_path_5, 'LSTM')
         #       print((os.path.join(LSTM_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(LSTM_path_5, 'mae_per_procedure.txt'), 'r') as file:
                    mae_LSTM_5.append(float(file.readline()[48:54]))
                with open(os.path.join(LSTM_path_5, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_LSTM_5.append(float(file.readline()[44:50]))
                with open(os.path.join(LSTM_path_5, 'time.txt'), 'r') as file:
                    traintime_LSTM_5.append(float(file.readline()[17:23]))
                    testtime_LSTM_5.append(float(file.readline()[16:22]))                    
                                   
                ############   LSTM_FCN  ########################
                LSTM_FCN_path_5 = os.path.join(fold_path_5, 'LSTM_FCN')
         #       print((os.path.join(LSTM_FCN_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(LSTM_FCN_path_5, 'mae_per_procedure.txt'), 'r') as file:
                    mae_LSTM_FCN_5.append(float(file.readline()[48:54]))
                with open(os.path.join(LSTM_FCN_path_5, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_LSTM_FCN_5.append(float(file.readline()[44:50]))
                with open(os.path.join(LSTM_FCN_path_5, 'time.txt'), 'r') as file:
                    traintime_LSTM_FCN_5.append(float(file.readline()[17:23]))
                    testtime_LSTM_FCN_5.append(float(file.readline()[16:22]))
                                                  
                ############   transformer  ########################
                transformer_path_5 = os.path.join(fold_path_5, 'transformer')
         #       print((os.path.join(transformer_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(transformer_path_5, 'mae_per_procedure.txt'), 'r') as file:
                    mae_transformer_5.append(float(file.readline()[48:54]))
                with open(os.path.join(transformer_path_5, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_transformer_5.append(float(file.readline()[44:50]))          
                with open(os.path.join(transformer_path_5, 'time.txt'), 'r') as file:
                    traintime_transformer_5.append(float(file.readline()[17:23]))
                    testtime_transformer_5.append(float(file.readline()[16:22]))
                    
               ############   LSTM_Attention  ########################
                LSTM_Attention_path_5 = os.path.join(fold_path_5, 'LSTM_Attention')
         #       print((os.path.join(LSTM_Attention_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(LSTM_Attention_path_5, 'mae_per_procedure.txt'), 'r') as file:
                    mae_LSTM_Attention_5.append(float(file.readline()[48:54]))
                with open(os.path.join(LSTM_Attention_path_5, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_LSTM_Attention_5.append(float(file.readline()[44:50]))                
                with open(os.path.join(LSTM_Attention_path_5, 'time.txt'), 'r') as file:
                    traintime_LSTM_Attention_5.append(float(file.readline()[17:23]))
                    testtime_LSTM_Attention_5.append(float(file.readline()[16:22]))                    
                    
                ############   Ensemble  ########################
                Ensemble_path_5 = os.path.join(fold_path_5, 'Ensemble')
         #       print((os.path.join(Ensemble_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(Ensemble_path_5, 'mae_per_procedure.txt'), 'r') as file:
                    mae_Ensemble_5.append(float(file.readline()[48:54]))
                with open(os.path.join(Ensemble_path_5, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_Ensemble_5.append(float(file.readline()[44:50]))       
            
        # Calculate the average accuracy number for the trial
        average_mae_INCEPTIONTIME_5.append(np.mean(mae_INCEPTIONTIME_5))
        average_SMAPE_INCEPTIONTIME_5.append(np.mean(SMAPE_INCEPTIONTIME_5))
        average_traintime_INCEPTIONTIME_5.append(np.mean(traintime_INCEPTIONTIME_5))
        average_testtime_INCEPTIONTIME_5.append(np.mean(testtime_INCEPTIONTIME_5))
        
        
        average_mae_LSTM_5.append(np.mean(mae_LSTM_5))
        average_SMAPE_LSTM_5.append(np.mean(SMAPE_LSTM_5))
        average_traintime_LSTM_5.append(np.mean(traintime_LSTM_5))
        average_testtime_LSTM_5.append(np.mean(testtime_LSTM_5))
        
        average_mae_LSTM_FCN_5.append(np.mean(mae_LSTM_FCN_5))
        average_SMAPE_LSTM_FCN_5.append(np.mean(SMAPE_LSTM_FCN_5))
        average_traintime_LSTM_FCN_5.append(np.mean(traintime_LSTM_FCN_5))
        average_testtime_LSTM_FCN_5.append(np.mean(testtime_LSTM_FCN_5))
        
        average_mae_transformer_5.append(np.mean(mae_transformer_5))
        average_SMAPE_transformer_5.append(np.mean(SMAPE_transformer_5))
        average_traintime_transformer_5.append(np.mean(traintime_transformer_5))
        average_testtime_transformer_5.append(np.mean(testtime_transformer_5))
        
        average_mae_LSTM_Attention_5.append(np.mean(mae_LSTM_Attention_5))
        average_SMAPE_LSTM_Attention_5.append(np.mean(SMAPE_LSTM_Attention_5))
        average_traintime_LSTM_Attention_5.append(np.mean(traintime_LSTM_Attention_5))
        average_testtime_LSTM_Attention_5.append(np.mean(testtime_LSTM_Attention_5))
                
        average_mae_Ensemble_5.append(np.mean(mae_Ensemble_5))
        average_SMAPE_Ensemble_5.append(np.mean(SMAPE_Ensemble_5))
        
        
    mae_list_5=[np.mean(np.array(average_mae_INCEPTIONTIME_5)),np.mean(np.array(average_mae_LSTM_5)),np.mean(np.array(average_mae_LSTM_FCN_5)),np.mean(np.array(average_mae_transformer_5))
                ,np.mean(np.array(average_mae_LSTM_Attention_5)),np.mean(np.array(average_mae_Ensemble_5))]
    smape_list_5=[np.mean(np.array(average_SMAPE_INCEPTIONTIME_5)),np.mean(np.array(average_SMAPE_LSTM_5)),np.mean(np.array(average_SMAPE_LSTM_FCN_5)),np.mean(np.array(average_SMAPE_transformer_5))
                  ,np.mean(np.array(average_SMAPE_LSTM_Attention_5)),np.mean(np.array(average_SMAPE_Ensemble_5))]
    traintime_list_5=[np.mean(np.array(average_traintime_INCEPTIONTIME_5)),np.mean(np.array(average_traintime_LSTM_5)),np.mean(np.array(average_traintime_LSTM_FCN_5)),np.mean(np.array(average_traintime_transformer_5)),np.mean(np.array(average_traintime_LSTM_Attention_5)),0]
    testtime_list_5=[np.mean(np.array(average_testtime_INCEPTIONTIME_5)),np.mean(np.array(average_testtime_LSTM_5)),np.mean(np.array(average_testtime_LSTM_FCN_5)),np.mean(np.array(average_testtime_transformer_5)),np.mean(np.array(average_testtime_LSTM_Attention_5)),0]
    traintime_list_5[5]=sum(traintime_list_5[0:4])
    testtime_list_5[5]=sum(testtime_list_5[0:4])
    
    results_5=pd.DataFrame({'Model': ['InceptionTime','LSTM','LSTM_FCN','Transformer','LSTM_Attention','Ensemble'],'MAE':mae_list_5,'SMAPE':smape_list_5,'Training time':traintime_list_5,'Testing time': testtime_list_5})

##################### 10 seconds ################################################
# Initialize lists to store accuracy numbers for each trial
average_mae_INCEPTIONTIME_10 = []
average_SMAPE_INCEPTIONTIME_10= []
average_traintime_INCEPTIONTIME_10=[]
average_testtime_INCEPTIONTIME_10=[]


average_mae_LSTM_10 = []
average_SMAPE_LSTM_10= []
average_traintime_LSTM_10=[]
average_testtime_LSTM_10=[]

average_mae_LSTM_FCN_10 = []
average_SMAPE_LSTM_FCN_10= []
average_traintime_LSTM_FCN_10=[]
average_testtime_LSTM_FCN_10=[]

average_mae_transformer_10 = []
average_SMAPE_transformer_10= []
average_traintime_transformer_10=[]
average_testtime_transformer_10=[]

average_mae_LSTM_Attention_10 = []
average_SMAPE_LSTM_Attention_10= []
average_traintime_LSTM_Attention_10=[]
average_testtime_LSTM_Attention_10=[]

average_mae_Ensemble_10 = []
average_SMAPE_Ensemble_10= []
# Loop through each trial folder
for trial_folder in os.listdir(main_folder):
    
    if trial_folder.startswith('trial_') and trial_folder.endswith(str(len(os.listdir(main_folder))-2))==False:# and int(trial_folder[-2:])!=(len(os.listdir(main_folder))-1):
        
        trial_path = os.path.join(main_folder, f'{trial_folder}')
        # Loop through each fold folder
        reduce_dim_path_10=os.path.join(trial_path, 'reduce_dim_10')
        
        mae_INCEPTIONTIME_10 = []
        SMAPE_INCEPTIONTIME_10= []
        traintime_INCEPTIONTIME_10=[]
        testtime_INCEPTIONTIME_10=[]
        
        mae_LSTM_10 = []
        SMAPE_LSTM_10= []
        traintime_LSTM_10=[]
        testtime_LSTM_10=[]
        
        mae_LSTM_FCN_10 = []
        SMAPE_LSTM_FCN_10= []
        traintime_LSTM_FCN_10=[]
        testtime_LSTM_FCN_10=[]
        
        mae_transformer_10 = []
        SMAPE_transformer_10= []
        traintime_transformer_10=[]
        testtime_transformer_10=[]
        
        mae_LSTM_Attention_10 = []
        SMAPE_LSTM_Attention_10= []
        traintime_LSTM_Attention_10=[]
        testtime_LSTM_Attention_10=[]
        
        mae_Ensemble_10 = []
        SMAPE_Ensemble_10= []
        for fold_folder in os.listdir(reduce_dim_path_10):
            if fold_folder.startswith('fold_'):
                fold_path_10=os.path.join(reduce_dim_path_10, fold_folder)
                
                ############   INCEPTION  ########################
                INCEPTIONTIME_path_10 = os.path.join(fold_path_10, 'INCEPTIONTIME')
         #       print((os.path.join(INCEPTIONTIME_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(INCEPTIONTIME_path_10, 'mae_per_procedure.txt'), 'r') as file:
                    mae_INCEPTIONTIME_10.append(float(file.readline()[48:54]))
                with open(os.path.join(INCEPTIONTIME_path_10, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_INCEPTIONTIME_10.append(float(file.readline()[44:50]))
                with open(os.path.join(INCEPTIONTIME_path_10, 'time.txt'), 'r') as file:
                    traintime_INCEPTIONTIME_10.append(float(file.readline()[17:23]))
                    testtime_INCEPTIONTIME_10.append(float(file.readline()[16:22]))
                                        
                
                ############   LSTM  ########################
                LSTM_path_10 = os.path.join(fold_path_10, 'LSTM')
         #       print((os.path.join(LSTM_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(LSTM_path_10, 'mae_per_procedure.txt'), 'r') as file:
                    mae_LSTM_10.append(float(file.readline()[48:54]))
                with open(os.path.join(LSTM_path_10, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_LSTM_10.append(float(file.readline()[44:50]))
                with open(os.path.join(LSTM_path_10, 'time.txt'), 'r') as file:
                    traintime_LSTM_10.append(float(file.readline()[17:23]))
                    testtime_LSTM_10.append(float(file.readline()[16:22]))                    
                                   
                ############   LSTM_FCN  ########################
                LSTM_FCN_path_10 = os.path.join(fold_path_10, 'LSTM_FCN')
         #       print((os.path.join(LSTM_FCN_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(LSTM_FCN_path_10, 'mae_per_procedure.txt'), 'r') as file:
                    mae_LSTM_FCN_10.append(float(file.readline()[48:54]))
                with open(os.path.join(LSTM_FCN_path_10, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_LSTM_FCN_10.append(float(file.readline()[44:50]))
                with open(os.path.join(LSTM_FCN_path_10, 'time.txt'), 'r') as file:
                    traintime_LSTM_FCN_10.append(float(file.readline()[17:23]))
                    testtime_LSTM_FCN_10.append(float(file.readline()[16:22]))
                                                  
                ############   transformer  ########################
                transformer_path_10 = os.path.join(fold_path_10, 'transformer')
         #       print((os.path.join(transformer_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(transformer_path_10, 'mae_per_procedure.txt'), 'r') as file:
                    mae_transformer_10.append(float(file.readline()[48:54]))
                with open(os.path.join(transformer_path_10, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_transformer_10.append(float(file.readline()[44:50]))          
                with open(os.path.join(transformer_path_10, 'time.txt'), 'r') as file:
                    traintime_transformer_10.append(float(file.readline()[17:23]))
                    testtime_transformer_10.append(float(file.readline()[16:22]))
                    
               ############   LSTM_Attention  ########################
                LSTM_Attention_path_10 = os.path.join(fold_path_10, 'LSTM_Attention')
         #       print((os.path.join(LSTM_Attention_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(LSTM_Attention_path_10, 'mae_per_procedure.txt'), 'r') as file:
                    mae_LSTM_Attention_10.append(float(file.readline()[48:54]))
                with open(os.path.join(LSTM_Attention_path_10, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_LSTM_Attention_10.append(float(file.readline()[44:50]))                
                with open(os.path.join(LSTM_Attention_path_10, 'time.txt'), 'r') as file:
                    traintime_LSTM_Attention_10.append(float(file.readline()[17:23]))
                    testtime_LSTM_Attention_10.append(float(file.readline()[16:22]))                    
                    
                ############   Ensemble  ########################
                Ensemble_path_10 = os.path.join(fold_path_10, 'Ensemble')
         #       print((os.path.join(Ensemble_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(Ensemble_path_10, 'mae_per_procedure.txt'), 'r') as file:
                    mae_Ensemble_10.append(float(file.readline()[48:54]))
                with open(os.path.join(Ensemble_path_10, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_Ensemble_10.append(float(file.readline()[44:50]))           
        # Calculate the average accuracy number for the trial
        average_mae_INCEPTIONTIME_10.append(np.mean(mae_INCEPTIONTIME_10))
        average_SMAPE_INCEPTIONTIME_10.append(np.mean(SMAPE_INCEPTIONTIME_10))
        average_traintime_INCEPTIONTIME_10.append(np.mean(traintime_INCEPTIONTIME_10))
        average_testtime_INCEPTIONTIME_10.append(np.mean(testtime_INCEPTIONTIME_10))
        
        
        average_mae_LSTM_10.append(np.mean(mae_LSTM_10))
        average_SMAPE_LSTM_10.append(np.mean(SMAPE_LSTM_10))
        average_traintime_LSTM_10.append(np.mean(traintime_LSTM_10))
        average_testtime_LSTM_10.append(np.mean(testtime_LSTM_10))
        
        average_mae_LSTM_FCN_10.append(np.mean(mae_LSTM_FCN_10))
        average_SMAPE_LSTM_FCN_10.append(np.mean(SMAPE_LSTM_FCN_10))
        average_traintime_LSTM_FCN_10.append(np.mean(traintime_LSTM_FCN_10))
        average_testtime_LSTM_FCN_10.append(np.mean(testtime_LSTM_FCN_10))
        
        average_mae_transformer_10.append(np.mean(mae_transformer_10))
        average_SMAPE_transformer_10.append(np.mean(SMAPE_transformer_10))
        average_traintime_transformer_10.append(np.mean(traintime_transformer_10))
        average_testtime_transformer_10.append(np.mean(testtime_transformer_10))
        
        average_mae_LSTM_Attention_10.append(np.mean(mae_LSTM_Attention_10))
        average_SMAPE_LSTM_Attention_10.append(np.mean(SMAPE_LSTM_Attention_10))
        average_traintime_LSTM_Attention_10.append(np.mean(traintime_LSTM_Attention_10))
        average_testtime_LSTM_Attention_10.append(np.mean(testtime_LSTM_Attention_10))
                
        average_mae_Ensemble_10.append(np.mean(mae_Ensemble_10))
        average_SMAPE_Ensemble_10.append(np.mean(SMAPE_Ensemble_10))
        
        
    mae_list_10=[np.mean(np.array(average_mae_INCEPTIONTIME_10)),np.mean(np.array(average_mae_LSTM_10)),np.mean(np.array(average_mae_LSTM_FCN_10)),np.mean(np.array(average_mae_transformer_10))
                ,np.mean(np.array(average_mae_LSTM_Attention_10)),np.mean(np.array(average_mae_Ensemble_10))]
    smape_list_10=[np.mean(np.array(average_SMAPE_INCEPTIONTIME_10)),np.mean(np.array(average_SMAPE_LSTM_10)),np.mean(np.array(average_SMAPE_LSTM_FCN_10)),np.mean(np.array(average_SMAPE_transformer_10))
                  ,np.mean(np.array(average_SMAPE_LSTM_Attention_10)),np.mean(np.array(average_SMAPE_Ensemble_10))]
    traintime_list_10=[np.mean(np.array(average_traintime_INCEPTIONTIME_10)),np.mean(np.array(average_traintime_LSTM_10)),np.mean(np.array(average_traintime_LSTM_FCN_10)),np.mean(np.array(average_traintime_transformer_10)),np.mean(np.array(average_traintime_LSTM_Attention_10)),0]
    testtime_list_10=[np.mean(np.array(average_testtime_INCEPTIONTIME_10)),np.mean(np.array(average_testtime_LSTM_10)),np.mean(np.array(average_testtime_LSTM_FCN_10)),np.mean(np.array(average_testtime_transformer_10)),np.mean(np.array(average_testtime_LSTM_Attention_10)),0]
    traintime_list_10[5]=sum(traintime_list_10[0:4])
    testtime_list_10[5]=sum(testtime_list_10[0:4])
    
    results_10=pd.DataFrame({'Model': ['InceptionTime','LSTM','LSTM_FCN','Transformer','LSTM_Attention','Ensemble'],'MAE':mae_list_10,'SMAPE':smape_list_10,'Training time':traintime_list_10,'Testing time': testtime_list_10})



##################### 30 seconds ################################################
# Initialize lists to store accuracy numbers for each trial
average_mae_INCEPTIONTIME_30 = []
average_SMAPE_INCEPTIONTIME_30= []
average_traintime_INCEPTIONTIME_30=[]
average_testtime_INCEPTIONTIME_30=[]


average_mae_LSTM_30 = []
average_SMAPE_LSTM_30= []
average_traintime_LSTM_30=[]
average_testtime_LSTM_30=[]

average_mae_LSTM_FCN_30 = []
average_SMAPE_LSTM_FCN_30= []
average_traintime_LSTM_FCN_30=[]
average_testtime_LSTM_FCN_30=[]

average_mae_transformer_30 = []
average_SMAPE_transformer_30= []
average_traintime_transformer_30=[]
average_testtime_transformer_30=[]

average_mae_LSTM_Attention_30 = []
average_SMAPE_LSTM_Attention_30= []
average_traintime_LSTM_Attention_30=[]
average_testtime_LSTM_Attention_30=[]

average_mae_Ensemble_30 = []
average_SMAPE_Ensemble_30= []
# Loop through each trial folder
for trial_folder in os.listdir(main_folder):
    
    if trial_folder.startswith('trial_') and trial_folder.endswith(str(len(os.listdir(main_folder))-2))==False:# and int(trial_folder[-2:])!=(len(os.listdir(main_folder))-1):
        
        trial_path = os.path.join(main_folder, f'{trial_folder}')
        # Loop through each fold folder
        reduce_dim_path_30=os.path.join(trial_path, 'reduce_dim_30')
        
        mae_INCEPTIONTIME_30 = []
        SMAPE_INCEPTIONTIME_30= []
        traintime_INCEPTIONTIME_30=[]
        testtime_INCEPTIONTIME_30=[]
        
        mae_LSTM_30 = []
        SMAPE_LSTM_30= []
        traintime_LSTM_30=[]
        testtime_LSTM_30=[]
        
        mae_LSTM_FCN_30 = []
        SMAPE_LSTM_FCN_30= []
        traintime_LSTM_FCN_30=[]
        testtime_LSTM_FCN_30=[]
        
        mae_transformer_30 = []
        SMAPE_transformer_30= []
        traintime_transformer_30=[]
        testtime_transformer_30=[]
        
        mae_LSTM_Attention_30 = []
        SMAPE_LSTM_Attention_30= []
        traintime_LSTM_Attention_30=[]
        testtime_LSTM_Attention_30=[]
        
        mae_Ensemble_30 = []
        SMAPE_Ensemble_30= []
        
        for fold_folder in os.listdir(reduce_dim_path_30):
            if fold_folder.startswith('fold_'):
                fold_path_30=os.path.join(reduce_dim_path_30, fold_folder)
                
                ############   INCEPTION  ########################
                INCEPTIONTIME_path_30 = os.path.join(fold_path_30, 'INCEPTIONTIME')
         #       print((os.path.join(INCEPTIONTIME_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(INCEPTIONTIME_path_30, 'mae_per_procedure.txt'), 'r') as file:
                    mae_INCEPTIONTIME_30.append(float(file.readline()[48:54]))
                with open(os.path.join(INCEPTIONTIME_path_30, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_INCEPTIONTIME_30.append(float(file.readline()[44:50]))
                with open(os.path.join(INCEPTIONTIME_path_30, 'time.txt'), 'r') as file:
                    traintime_INCEPTIONTIME_30.append(float(file.readline()[17:23]))
                    testtime_INCEPTIONTIME_30.append(float(file.readline()[16:22]))
                                        
                
                ############   LSTM  ########################
                LSTM_path_30 = os.path.join(fold_path_30, 'LSTM')
         #       print((os.path.join(LSTM_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(LSTM_path_30, 'mae_per_procedure.txt'), 'r') as file:
                    mae_LSTM_30.append(float(file.readline()[48:54]))
                with open(os.path.join(LSTM_path_30, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_LSTM_30.append(float(file.readline()[44:50]))
                with open(os.path.join(LSTM_path_30, 'time.txt'), 'r') as file:
                    traintime_LSTM_30.append(float(file.readline()[17:23]))
                    testtime_LSTM_30.append(float(file.readline()[16:22]))                    
                                   
                ############   LSTM_FCN  ########################
                LSTM_FCN_path_30 = os.path.join(fold_path_30, 'LSTM_FCN')
         #       print((os.path.join(LSTM_FCN_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(LSTM_FCN_path_30, 'mae_per_procedure.txt'), 'r') as file:
                    mae_LSTM_FCN_30.append(float(file.readline()[48:54]))
                with open(os.path.join(LSTM_FCN_path_30, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_LSTM_FCN_30.append(float(file.readline()[44:50]))
                with open(os.path.join(LSTM_FCN_path_30, 'time.txt'), 'r') as file:
                    traintime_LSTM_FCN_30.append(float(file.readline()[17:23]))
                    testtime_LSTM_FCN_30.append(float(file.readline()[16:22]))
                                                  
                ############   transformer  ########################
                transformer_path_30 = os.path.join(fold_path_30, 'transformer')
         #       print((os.path.join(transformer_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(transformer_path_30, 'mae_per_procedure.txt'), 'r') as file:
                    mae_transformer_30.append(float(file.readline()[48:54]))
                with open(os.path.join(transformer_path_30, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_transformer_30.append(float(file.readline()[44:50]))          
                with open(os.path.join(transformer_path_30, 'time.txt'), 'r') as file:
                    traintime_transformer_30.append(float(file.readline()[17:23]))
                    testtime_transformer_30.append(float(file.readline()[16:22]))
                    
               ############   LSTM_Attention  ########################
                LSTM_Attention_path_30 = os.path.join(fold_path_30, 'LSTM_Attention')
         #       print((os.path.join(LSTM_Attention_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(LSTM_Attention_path_30, 'mae_per_procedure.txt'), 'r') as file:
                    mae_LSTM_Attention_30.append(float(file.readline()[48:54]))
                with open(os.path.join(LSTM_Attention_path_30, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_LSTM_Attention_30.append(float(file.readline()[44:50]))                
                with open(os.path.join(LSTM_Attention_path_30, 'time.txt'), 'r') as file:
                    traintime_LSTM_Attention_30.append(float(file.readline()[17:23]))
                    testtime_LSTM_Attention_30.append(float(file.readline()[16:22]))                    
                    
                ############   Ensemble  ########################
                Ensemble_path_30 = os.path.join(fold_path_30, 'Ensemble')
         #       print((os.path.join(Ensemble_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(Ensemble_path_30, 'mae_per_procedure.txt'), 'r') as file:
                    mae_Ensemble_30.append(float(file.readline()[48:54]))
                with open(os.path.join(Ensemble_path_30, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_Ensemble_30.append(float(file.readline()[44:50]))                
            
        # Calculate the average accuracy number for the trial
        average_mae_INCEPTIONTIME_30.append(np.mean(mae_INCEPTIONTIME_30))
        average_SMAPE_INCEPTIONTIME_30.append(np.mean(SMAPE_INCEPTIONTIME_30))
        average_traintime_INCEPTIONTIME_30.append(np.mean(traintime_INCEPTIONTIME_30))
        average_testtime_INCEPTIONTIME_30.append(np.mean(testtime_INCEPTIONTIME_30))
        
        
        average_mae_LSTM_30.append(np.mean(mae_LSTM_30))
        average_SMAPE_LSTM_30.append(np.mean(SMAPE_LSTM_30))
        average_traintime_LSTM_30.append(np.mean(traintime_LSTM_30))
        average_testtime_LSTM_30.append(np.mean(testtime_LSTM_30))
        
        average_mae_LSTM_FCN_30.append(np.mean(mae_LSTM_FCN_30))
        average_SMAPE_LSTM_FCN_30.append(np.mean(SMAPE_LSTM_FCN_30))
        average_traintime_LSTM_FCN_30.append(np.mean(traintime_LSTM_FCN_30))
        average_testtime_LSTM_FCN_30.append(np.mean(testtime_LSTM_FCN_30))
        
        average_mae_transformer_30.append(np.mean(mae_transformer_30))
        average_SMAPE_transformer_30.append(np.mean(SMAPE_transformer_30))
        average_traintime_transformer_30.append(np.mean(traintime_transformer_30))
        average_testtime_transformer_30.append(np.mean(testtime_transformer_30))
        
        average_mae_LSTM_Attention_30.append(np.mean(mae_LSTM_Attention_30))
        average_SMAPE_LSTM_Attention_30.append(np.mean(SMAPE_LSTM_Attention_30))
        average_traintime_LSTM_Attention_30.append(np.mean(traintime_LSTM_Attention_30))
        average_testtime_LSTM_Attention_30.append(np.mean(testtime_LSTM_Attention_30))
                
        average_mae_Ensemble_30.append(np.mean(mae_Ensemble_30))
        average_SMAPE_Ensemble_30.append(np.mean(SMAPE_Ensemble_30))
        
        
    mae_list_30=[np.mean(np.array(average_mae_INCEPTIONTIME_30)),np.mean(np.array(average_mae_LSTM_30)),np.mean(np.array(average_mae_LSTM_FCN_30)),np.mean(np.array(average_mae_transformer_30))
                ,np.mean(np.array(average_mae_LSTM_Attention_30)),np.mean(np.array(average_mae_Ensemble_30))]
    smape_list_30=[np.mean(np.array(average_SMAPE_INCEPTIONTIME_30)),np.mean(np.array(average_SMAPE_LSTM_30)),np.mean(np.array(average_SMAPE_LSTM_FCN_30)),np.mean(np.array(average_SMAPE_transformer_30))
                  ,np.mean(np.array(average_SMAPE_LSTM_Attention_30)),np.mean(np.array(average_SMAPE_Ensemble_30))]
    traintime_list_30=[np.mean(np.array(average_traintime_INCEPTIONTIME_30)),np.mean(np.array(average_traintime_LSTM_30)),np.mean(np.array(average_traintime_LSTM_FCN_30)),np.mean(np.array(average_traintime_transformer_30)),np.mean(np.array(average_traintime_LSTM_Attention_30)),0]
    testtime_list_30=[np.mean(np.array(average_testtime_INCEPTIONTIME_30)),np.mean(np.array(average_testtime_LSTM_30)),np.mean(np.array(average_testtime_LSTM_FCN_30)),np.mean(np.array(average_testtime_transformer_30)),np.mean(np.array(average_testtime_LSTM_Attention_30)),0]
    traintime_list_30[5]=sum(traintime_list_30[0:4])
    testtime_list_30[5]=sum(testtime_list_30[0:4])
    
    results_30=pd.DataFrame({'Model': ['InceptionTime','LSTM','LSTM_FCN','Transformer','LSTM_Attention','Ensemble'],'MAE':mae_list_30,'SMAPE':smape_list_30,'Training time':traintime_list_30,'Testing time': testtime_list_30})

##################### 60 seconds ################################################
# Initialize lists to store accuracy numbers for each trial
average_mae_INCEPTIONTIME_60 = []
average_SMAPE_INCEPTIONTIME_60= []
average_traintime_INCEPTIONTIME_60=[]
average_testtime_INCEPTIONTIME_60=[]


average_mae_LSTM_60 = []
average_SMAPE_LSTM_60= []
average_traintime_LSTM_60=[]
average_testtime_LSTM_60=[]

average_mae_LSTM_FCN_60 = []
average_SMAPE_LSTM_FCN_60= []
average_traintime_LSTM_FCN_60=[]
average_testtime_LSTM_FCN_60=[]

average_mae_transformer_60 = []
average_SMAPE_transformer_60= []
average_traintime_transformer_60=[]
average_testtime_transformer_60=[]

average_mae_LSTM_Attention_60 = []
average_SMAPE_LSTM_Attention_60= []
average_traintime_LSTM_Attention_60=[]
average_testtime_LSTM_Attention_60=[]

average_mae_Ensemble_60 = []
average_SMAPE_Ensemble_60= []
# Loop through each trial folder
for trial_folder in os.listdir(main_folder):
    
    if trial_folder.startswith('trial_') and trial_folder.endswith(str(len(os.listdir(main_folder))-2))==False:# and int(trial_folder[-2:])!=(len(os.listdir(main_folder))-1):
        
        trial_path = os.path.join(main_folder, f'{trial_folder}')
        # Loop through each fold folder
        reduce_dim_path_60=os.path.join(trial_path, 'reduce_dim_60')
        
        mae_INCEPTIONTIME_60 = []
        SMAPE_INCEPTIONTIME_60= []
        traintime_INCEPTIONTIME_60=[]
        testtime_INCEPTIONTIME_60=[]
        
        mae_LSTM_60 = []
        SMAPE_LSTM_60= []
        traintime_LSTM_60=[]
        testtime_LSTM_60=[]
        
        mae_LSTM_FCN_60 = []
        SMAPE_LSTM_FCN_60= []
        traintime_LSTM_FCN_60=[]
        testtime_LSTM_FCN_60=[]
        
        mae_transformer_60 = []
        SMAPE_transformer_60= []
        traintime_transformer_60=[]
        testtime_transformer_60=[]
        
        mae_LSTM_Attention_60 = []
        SMAPE_LSTM_Attention_60= []
        traintime_LSTM_Attention_60=[]
        testtime_LSTM_Attention_60=[]
        
        mae_Ensemble_60 = []
        SMAPE_Ensemble_60= []
        
        for fold_folder in os.listdir(reduce_dim_path_60):
            if fold_folder.startswith('fold_'):
                fold_path_60=os.path.join(reduce_dim_path_60, fold_folder)
                
                ############   INCEPTION  ########################
                INCEPTIONTIME_path_60 = os.path.join(fold_path_60, 'INCEPTIONTIME')
         #       print((os.path.join(INCEPTIONTIME_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(INCEPTIONTIME_path_60, 'mae_per_procedure.txt'), 'r') as file:
                    mae_INCEPTIONTIME_60.append(float(file.readline()[48:54]))
                with open(os.path.join(INCEPTIONTIME_path_60, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_INCEPTIONTIME_60.append(float(file.readline()[44:50]))
                with open(os.path.join(INCEPTIONTIME_path_60, 'time.txt'), 'r') as file:
                    traintime_INCEPTIONTIME_60.append(float(file.readline()[17:23]))
                    testtime_INCEPTIONTIME_60.append(float(file.readline()[16:22]))
                                        
                
                ############   LSTM  ########################
                LSTM_path_60 = os.path.join(fold_path_60, 'LSTM')
         #       print((os.path.join(LSTM_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(LSTM_path_60, 'mae_per_procedure.txt'), 'r') as file:
                    mae_LSTM_60.append(float(file.readline()[48:54]))
                with open(os.path.join(LSTM_path_60, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_LSTM_60.append(float(file.readline()[44:50]))
                with open(os.path.join(LSTM_path_60, 'time.txt'), 'r') as file:
                    traintime_LSTM_60.append(float(file.readline()[17:23]))
                    testtime_LSTM_60.append(float(file.readline()[16:22]))                    
                                   
                ############   LSTM_FCN  ########################
                LSTM_FCN_path_60 = os.path.join(fold_path_60, 'LSTM_FCN')
         #       print((os.path.join(LSTM_FCN_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(LSTM_FCN_path_60, 'mae_per_procedure.txt'), 'r') as file:
                    mae_LSTM_FCN_60.append(float(file.readline()[48:54]))
                with open(os.path.join(LSTM_FCN_path_60, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_LSTM_FCN_60.append(float(file.readline()[44:50]))
                with open(os.path.join(LSTM_FCN_path_60, 'time.txt'), 'r') as file:
                    traintime_LSTM_FCN_60.append(float(file.readline()[17:23]))
                    testtime_LSTM_FCN_60.append(float(file.readline()[16:22]))
                                                  
                ############   transformer  ########################
                transformer_path_60 = os.path.join(fold_path_60, 'transformer')
         #       print((os.path.join(transformer_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(transformer_path_60, 'mae_per_procedure.txt'), 'r') as file:
                    mae_transformer_60.append(float(file.readline()[48:54]))
                with open(os.path.join(transformer_path_60, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_transformer_60.append(float(file.readline()[44:50]))          
                with open(os.path.join(transformer_path_60, 'time.txt'), 'r') as file:
                    traintime_transformer_60.append(float(file.readline()[17:23]))
                    testtime_transformer_60.append(float(file.readline()[16:22]))
                    
               ############   LSTM_Attention  ########################
                LSTM_Attention_path_60 = os.path.join(fold_path_60, 'LSTM_Attention')
         #       print((os.path.join(LSTM_Attention_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(LSTM_Attention_path_60, 'mae_per_procedure.txt'), 'r') as file:
                    mae_LSTM_Attention_60.append(float(file.readline()[48:54]))
                with open(os.path.join(LSTM_Attention_path_60, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_LSTM_Attention_60.append(float(file.readline()[44:50]))                
                with open(os.path.join(LSTM_Attention_path_60, 'time.txt'), 'r') as file:
                    traintime_LSTM_Attention_60.append(float(file.readline()[17:23]))
                    testtime_LSTM_Attention_60.append(float(file.readline()[16:22]))                    
                    
                ############   Ensemble  ########################
                Ensemble_path_60 = os.path.join(fold_path_60, 'Ensemble')
         #       print((os.path.join(Ensemble_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(Ensemble_path_60, 'mae_per_procedure.txt'), 'r') as file:
                    mae_Ensemble_60.append(float(file.readline()[48:54]))
                with open(os.path.join(Ensemble_path_60, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_Ensemble_60.append(float(file.readline()[44:50]))                
            
        # Calculate the average accuracy number for the trial
        average_mae_INCEPTIONTIME_60.append(np.mean(mae_INCEPTIONTIME_60))
        average_SMAPE_INCEPTIONTIME_60.append(np.mean(SMAPE_INCEPTIONTIME_60))
        average_traintime_INCEPTIONTIME_60.append(np.mean(traintime_INCEPTIONTIME_60))
        average_testtime_INCEPTIONTIME_60.append(np.mean(testtime_INCEPTIONTIME_60))
        
        
        average_mae_LSTM_60.append(np.mean(mae_LSTM_60))
        average_SMAPE_LSTM_60.append(np.mean(SMAPE_LSTM_60))
        average_traintime_LSTM_60.append(np.mean(traintime_LSTM_60))
        average_testtime_LSTM_60.append(np.mean(testtime_LSTM_60))
        
        average_mae_LSTM_FCN_60.append(np.mean(mae_LSTM_FCN_60))
        average_SMAPE_LSTM_FCN_60.append(np.mean(SMAPE_LSTM_FCN_60))
        average_traintime_LSTM_FCN_60.append(np.mean(traintime_LSTM_FCN_60))
        average_testtime_LSTM_FCN_60.append(np.mean(testtime_LSTM_FCN_60))
        
        average_mae_transformer_60.append(np.mean(mae_transformer_60))
        average_SMAPE_transformer_60.append(np.mean(SMAPE_transformer_60))
        average_traintime_transformer_60.append(np.mean(traintime_transformer_60))
        average_testtime_transformer_60.append(np.mean(testtime_transformer_60))
        
        average_mae_LSTM_Attention_60.append(np.mean(mae_LSTM_Attention_60))
        average_SMAPE_LSTM_Attention_60.append(np.mean(SMAPE_LSTM_Attention_60))
        average_traintime_LSTM_Attention_60.append(np.mean(traintime_LSTM_Attention_60))
        average_testtime_LSTM_Attention_60.append(np.mean(testtime_LSTM_Attention_60))
                
        average_mae_Ensemble_60.append(np.mean(mae_Ensemble_60))
        average_SMAPE_Ensemble_60.append(np.mean(SMAPE_Ensemble_60))
        
        
    mae_list_60=[np.mean(np.array(average_mae_INCEPTIONTIME_60)),np.mean(np.array(average_mae_LSTM_60)),np.mean(np.array(average_mae_LSTM_FCN_60)),np.mean(np.array(average_mae_transformer_60))
                ,np.mean(np.array(average_mae_LSTM_Attention_60)),np.mean(np.array(average_mae_Ensemble_60))]
    smape_list_60=[np.mean(np.array(average_SMAPE_INCEPTIONTIME_60)),np.mean(np.array(average_SMAPE_LSTM_60)),np.mean(np.array(average_SMAPE_LSTM_FCN_60)),np.mean(np.array(average_SMAPE_transformer_60))
                  ,np.mean(np.array(average_SMAPE_LSTM_Attention_60)),np.mean(np.array(average_SMAPE_Ensemble_60))]
    traintime_list_60=[np.mean(np.array(average_traintime_INCEPTIONTIME_60)),np.mean(np.array(average_traintime_LSTM_60)),np.mean(np.array(average_traintime_LSTM_FCN_60)),np.mean(np.array(average_traintime_transformer_60)),np.mean(np.array(average_traintime_LSTM_Attention_60)),0]
    testtime_list_60=[np.mean(np.array(average_testtime_INCEPTIONTIME_60)),np.mean(np.array(average_testtime_LSTM_60)),np.mean(np.array(average_testtime_LSTM_FCN_60)),np.mean(np.array(average_testtime_transformer_60)),np.mean(np.array(average_testtime_LSTM_Attention_60)),0]
    traintime_list_60[5]=sum(traintime_list_60[0:4])
    testtime_list_60[5]=sum(testtime_list_60[0:4])
    
    results_60=pd.DataFrame({'Model': ['InceptionTime','LSTM','LSTM_FCN','Transformer','LSTM_Attention','Ensemble'],'MAE':mae_list_60,'SMAPE':smape_list_60,'Training time':traintime_list_60,'Testing time': testtime_list_60})

##################### 120 seconds ################################################
# Initialize lists to store accuracy numbers for each trial
average_mae_INCEPTIONTIME_120 = []
average_SMAPE_INCEPTIONTIME_120= []
average_traintime_INCEPTIONTIME_120=[]
average_testtime_INCEPTIONTIME_120=[]


average_mae_LSTM_120 = []
average_SMAPE_LSTM_120= []
average_traintime_LSTM_120=[]
average_testtime_LSTM_120=[]

average_mae_LSTM_FCN_120 = []
average_SMAPE_LSTM_FCN_120= []
average_traintime_LSTM_FCN_120=[]
average_testtime_LSTM_FCN_120=[]

average_mae_transformer_120 = []
average_SMAPE_transformer_120= []
average_traintime_transformer_120=[]
average_testtime_transformer_120=[]

average_mae_LSTM_Attention_120 = []
average_SMAPE_LSTM_Attention_120= []
average_traintime_LSTM_Attention_120=[]
average_testtime_LSTM_Attention_120=[]

average_mae_Ensemble_120 = []
average_SMAPE_Ensemble_120= []
# Loop through each trial folder
for trial_folder in os.listdir(main_folder):
    
    if trial_folder.startswith('trial_') and trial_folder.endswith(str(len(os.listdir(main_folder))-2))==False:# and int(trial_folder[-2:])!=(len(os.listdir(main_folder))-1):
        
        trial_path = os.path.join(main_folder, f'{trial_folder}')
        # Loop through each fold folder
        reduce_dim_path_120=os.path.join(trial_path, 'reduce_dim_120')
        
        mae_INCEPTIONTIME_120 = []
        SMAPE_INCEPTIONTIME_120= []
        traintime_INCEPTIONTIME_120=[]
        testtime_INCEPTIONTIME_120=[]
        
        mae_LSTM_120 = []
        SMAPE_LSTM_120= []
        traintime_LSTM_120=[]
        testtime_LSTM_120=[]
        
        mae_LSTM_FCN_120 = []
        SMAPE_LSTM_FCN_120= []
        traintime_LSTM_FCN_120=[]
        testtime_LSTM_FCN_120=[]
        
        mae_transformer_120 = []
        SMAPE_transformer_120= []
        traintime_transformer_120=[]
        testtime_transformer_120=[]
        
        mae_LSTM_Attention_120 = []
        SMAPE_LSTM_Attention_120= []
        traintime_LSTM_Attention_120=[]
        testtime_LSTM_Attention_120=[]
        
        mae_Ensemble_120 = []
        SMAPE_Ensemble_120= []
        
        for fold_folder in os.listdir(reduce_dim_path_120):
            if fold_folder.startswith('fold_'):
                fold_path_120=os.path.join(reduce_dim_path_120, fold_folder)
                
                ############   INCEPTION  ########################
                INCEPTIONTIME_path_120 = os.path.join(fold_path_120, 'INCEPTIONTIME')
         #       print((os.path.join(INCEPTIONTIME_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(INCEPTIONTIME_path_120, 'mae_per_procedure.txt'), 'r') as file:
                    mae_INCEPTIONTIME_120.append(float(file.readline()[48:54]))
                with open(os.path.join(INCEPTIONTIME_path_120, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_INCEPTIONTIME_120.append(float(file.readline()[44:50]))
                with open(os.path.join(INCEPTIONTIME_path_120, 'time.txt'), 'r') as file:
                    traintime_INCEPTIONTIME_120.append(float(file.readline()[17:23]))
                    testtime_INCEPTIONTIME_120.append(float(file.readline()[16:22]))
                                        
                
                ############   LSTM  ########################
                LSTM_path_120 = os.path.join(fold_path_120, 'LSTM')
         #       print((os.path.join(LSTM_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(LSTM_path_120, 'mae_per_procedure.txt'), 'r') as file:
                    mae_LSTM_120.append(float(file.readline()[48:54]))
                with open(os.path.join(LSTM_path_120, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_LSTM_120.append(float(file.readline()[44:50]))
                with open(os.path.join(LSTM_path_120, 'time.txt'), 'r') as file:
                    traintime_LSTM_120.append(float(file.readline()[17:23]))
                    testtime_LSTM_120.append(float(file.readline()[16:22]))                    
                                   
                ############   LSTM_FCN  ########################
                LSTM_FCN_path_120 = os.path.join(fold_path_120, 'LSTM_FCN')
         #       print((os.path.join(LSTM_FCN_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(LSTM_FCN_path_120, 'mae_per_procedure.txt'), 'r') as file:
                    mae_LSTM_FCN_120.append(float(file.readline()[48:54]))
                with open(os.path.join(LSTM_FCN_path_120, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_LSTM_FCN_120.append(float(file.readline()[44:50]))
                with open(os.path.join(LSTM_FCN_path_120, 'time.txt'), 'r') as file:
                    traintime_LSTM_FCN_120.append(float(file.readline()[17:23]))
                    testtime_LSTM_FCN_120.append(float(file.readline()[16:22]))
                                                  
                ############   transformer  ########################
                transformer_path_120 = os.path.join(fold_path_120, 'transformer')
         #       print((os.path.join(transformer_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(transformer_path_120, 'mae_per_procedure.txt'), 'r') as file:
                    mae_transformer_120.append(float(file.readline()[48:54]))
                with open(os.path.join(transformer_path_120, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_transformer_120.append(float(file.readline()[44:50]))          
                with open(os.path.join(transformer_path_120, 'time.txt'), 'r') as file:
                    traintime_transformer_120.append(float(file.readline()[17:23]))
                    testtime_transformer_120.append(float(file.readline()[16:22]))
                    
               ############   LSTM_Attention  ########################
                LSTM_Attention_path_120 = os.path.join(fold_path_120, 'LSTM_Attention')
         #       print((os.path.join(LSTM_Attention_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(LSTM_Attention_path_120, 'mae_per_procedure.txt'), 'r') as file:
                    mae_LSTM_Attention_120.append(float(file.readline()[48:54]))
                with open(os.path.join(LSTM_Attention_path_120, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_LSTM_Attention_120.append(float(file.readline()[44:50]))                
                with open(os.path.join(LSTM_Attention_path_120, 'time.txt'), 'r') as file:
                    traintime_LSTM_Attention_120.append(float(file.readline()[17:23]))
                    testtime_LSTM_Attention_120.append(float(file.readline()[16:22]))                    
                    
                ############   Ensemble  ########################
                Ensemble_path_120 = os.path.join(fold_path_120, 'Ensemble')
         #       print((os.path.join(Ensemble_path, 'mae_per_procedure.txt')))
                # Read the accuracy number from the file
                with open(os.path.join(Ensemble_path_120, 'mae_per_procedure.txt'), 'r') as file:
                    mae_Ensemble_120.append(float(file.readline()[48:54]))
                with open(os.path.join(Ensemble_path_120, 'SMAPE_per_procedure.txt'), 'r') as file:
                    SMAPE_Ensemble_120.append(float(file.readline()[44:50]))                
            
        # Calculate the average accuracy number for the trial
        average_mae_INCEPTIONTIME_120.append(np.mean(mae_INCEPTIONTIME_120))
        average_SMAPE_INCEPTIONTIME_120.append(np.mean(SMAPE_INCEPTIONTIME_120))
        average_traintime_INCEPTIONTIME_120.append(np.mean(traintime_INCEPTIONTIME_120))
        average_testtime_INCEPTIONTIME_120.append(np.mean(testtime_INCEPTIONTIME_120))
        
        
        average_mae_LSTM_120.append(np.mean(mae_LSTM_120))
        average_SMAPE_LSTM_120.append(np.mean(SMAPE_LSTM_120))
        average_traintime_LSTM_120.append(np.mean(traintime_LSTM_120))
        average_testtime_LSTM_120.append(np.mean(testtime_LSTM_120))
        
        average_mae_LSTM_FCN_120.append(np.mean(mae_LSTM_FCN_120))
        average_SMAPE_LSTM_FCN_120.append(np.mean(SMAPE_LSTM_FCN_120))
        average_traintime_LSTM_FCN_120.append(np.mean(traintime_LSTM_FCN_120))
        average_testtime_LSTM_FCN_120.append(np.mean(testtime_LSTM_FCN_120))
        
        average_mae_transformer_120.append(np.mean(mae_transformer_120))
        average_SMAPE_transformer_120.append(np.mean(SMAPE_transformer_120))
        average_traintime_transformer_120.append(np.mean(traintime_transformer_120))
        average_testtime_transformer_120.append(np.mean(testtime_transformer_120))
        
        average_mae_LSTM_Attention_120.append(np.mean(mae_LSTM_Attention_120))
        average_SMAPE_LSTM_Attention_120.append(np.mean(SMAPE_LSTM_Attention_120))
        average_traintime_LSTM_Attention_120.append(np.mean(traintime_LSTM_Attention_120))
        average_testtime_LSTM_Attention_120.append(np.mean(testtime_LSTM_Attention_120))
                
        average_mae_Ensemble_120.append(np.mean(mae_Ensemble_120))
        average_SMAPE_Ensemble_120.append(np.mean(SMAPE_Ensemble_120))
        
        
    mae_list_120=[np.mean(np.array(average_mae_INCEPTIONTIME_120)),np.mean(np.array(average_mae_LSTM_120)),np.mean(np.array(average_mae_LSTM_FCN_120)),np.mean(np.array(average_mae_transformer_120))
                ,np.mean(np.array(average_mae_LSTM_Attention_120)),np.mean(np.array(average_mae_Ensemble_120))]
    smape_list_120=[np.mean(np.array(average_SMAPE_INCEPTIONTIME_120)),np.mean(np.array(average_SMAPE_LSTM_120)),np.mean(np.array(average_SMAPE_LSTM_FCN_120)),np.mean(np.array(average_SMAPE_transformer_120))
                  ,np.mean(np.array(average_SMAPE_LSTM_Attention_120)),np.mean(np.array(average_SMAPE_Ensemble_120))]
    traintime_list_120=[np.mean(np.array(average_traintime_INCEPTIONTIME_120)),np.mean(np.array(average_traintime_LSTM_120)),np.mean(np.array(average_traintime_LSTM_FCN_120)),np.mean(np.array(average_traintime_transformer_120)),np.mean(np.array(average_traintime_LSTM_Attention_120)),0]
    testtime_list_120=[np.mean(np.array(average_testtime_INCEPTIONTIME_120)),np.mean(np.array(average_testtime_LSTM_120)),np.mean(np.array(average_testtime_LSTM_FCN_120)),np.mean(np.array(average_testtime_transformer_120)),np.mean(np.array(average_testtime_LSTM_Attention_120)),0]
    traintime_list_120[5]=sum(traintime_list_120[0:4])
    testtime_list_120[5]=sum(testtime_list_120[0:4])

    results_120=pd.DataFrame({'Model': ['InceptionTime','LSTM','LSTM_FCN','Transformer','LSTM_Attention','Ensemble'],'MAE':mae_list_120,'SMAPE':smape_list_120,'Training time':traintime_list_120,'Testing time': testtime_list_120})

results_5=results_5.round(2)
results_10=results_10.round(2)
results_30=results_30.round(2)
results_60=results_60.round(2)
results_120=results_120.round(2)

with pd.ExcelWriter(r'C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Codes\MTSC\runs\runs_PAPER_20240429-121956\summary.xlsx') as writer:
    # Write each DataFrame to a separate sheet
    results_5.to_excel(writer, sheet_name='Sheet1', startrow=0, startcol=0, index=False)
    results_10.to_excel(writer, sheet_name='Sheet1', startrow=0, startcol=len(results_5.columns) + 1, index=False)
    results_30.to_excel(writer, sheet_name='Sheet1', startrow=0, startcol=len(results_10.columns)*2 + 2, index=False)
    results_60.to_excel(writer, sheet_name='Sheet1', startrow=0, startcol=len(results_30.columns)*3 + 3, index=False)
    results_120.to_excel(writer, sheet_name='Sheet1', startrow=0, startcol=len(results_60.columns)*4 + 4, index=False)

