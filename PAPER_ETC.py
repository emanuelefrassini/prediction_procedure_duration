#Models training for paper
######################### Structure ################################
#import pickle
import numpy as np
#import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#from tensorflow.keras.utils import to_categorical
#import tensorflow as tf
#from tensorflow.keras.losses import Loss,MeanAbsoluteError
#import keras
#from keras.layers import LSTM,Bidirectional, Dropout, Dense,Input, multiply, concatenate, Activation, Masking, Reshape,Conv1D, BatchNormalization, GlobalAveragePooling1D
from keras.models import save_model
#import tensorflow as tf
#from keras import layers
#import numpy as np
#from tensorflow.keras.callbacks import TensorBoard
#import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
#from sklearn.metrics import classification_report
import time
#from sklearn.preprocessing import MinMaxScaler
import os
import datetime
#from tensorflow.keras.callbacks import Callback
#from tensorboard.plugins.hparams import api as hp
#from utils.models import model_INCEPTIONTIME, model_LSTM, model_LSTM_Attention, model_LSTM_FCN, model_transformer,smape
from contextlib import redirect_stdout

##PARAMETERS
reduce_dim=[5,10,30,60,120]
NUM_FOLDS=3
EPOCHS=1000
#BATCH_SIZE=8

#Create special folder run
base_folder = r'C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Codes\MTSC\runs\\'

# Get the current date and time
current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# Path for the new folder
folder_path_inforun = os.path.join(base_folder, f'runs_PAPER_{current_datetime}')
# Info run save path
os.makedirs(folder_path_inforun, exist_ok=True)
##Data
#Load data
def load_data(file_path= r'C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Codes\MTSC\time_binary_regression\data\dataset.npz'):

    loaded_data = np.load(file_path)

    # Retrieve arrays X and Y from loaded data
    X = loaded_data['X']
    Y = loaded_data['Y']
    return X,Y
Xload,Yload=load_data()
#Repeat n times all this:
    #Set reduce dimension, per reduce_dim: 
for trial in range(91,100):
    folder_path_inforun2 = os.path.join(folder_path_inforun, f'trial_{trial}')
    # Info run save path
    os.makedirs(folder_path_inforun2, exist_ok=True)
    for REDUCE_DIM in reduce_dim: 
        X=Xload
        Y=Yload     
        X=X[:,0::REDUCE_DIM,:]
        Y=Y[:,0::REDUCE_DIM,:] 
        X_1=X.shape[1]
        X_2=X.shape[2]
        ATTENTION_UNITS=X_1
        
          # Path for the new folder
        folder_path_perdim = os.path.join(folder_path_inforun2, f'reduce_dim_{REDUCE_DIM}')
        # Info run save path
        os.makedirs(folder_path_perdim, exist_ok=True)
            
        # Experiment set up
        kfold = KFold(n_splits=NUM_FOLDS, shuffle=True)  
        
        fold = 0
        
        for train, test in kfold.split(X):
            from utils.models import model_INCEPTIONTIME, model_LSTM, model_LSTM_Attention, model_LSTM_FCN, model_transformer,smape
            start_fold = time.time()
            
            folder_path_perfold = os.path.join(folder_path_perdim, f'fold_{fold}')
            # Info run save path
            os.makedirs(folder_path_perfold, exist_ok=True)
            indices = {"train": train, "test": test}
      
      # Save indices into a single file
            file_path = os.path.join(folder_path_perfold, "Indices.npy")
            np.save(file_path, indices)
            
            fold += 1
            
            X_train = X[train, :, :]
            X_test = X[test, :, :]
        
            scaler_x = MinMaxScaler()
            X_train = scaler_x.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            X_test = scaler_x.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
         
            y_train = np.float32(Y[train])
            y_test = np.float32(Y[test])
        
            scaler_y = MinMaxScaler()
            y_train = scaler_y.fit_transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train.shape)
            y_test = scaler_y.transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)
            
            
         
            #Process data (function)
            ##Training
            #3fold split, per split train:
            model_LSTM,model_callbacks=model_LSTM(UNITS=128,DROPOUT_RATE=0.1,LEARNING_RATE=0.001)
            model_LSTM_Attention,model_callbacks=model_LSTM_Attention(UNITS=128,DROPOUT_RATE=0.1,LEARNING_RATE=0.001)
            model_LSTM_FCN,model_callbacks=model_LSTM_FCN(X_1,X_2,UNITS=256,DROPOUT_RATE=0.1,LEARNING_RATE=0.001)
            model_transformer,model_callbacks=model_transformer(X_1,X_2, LEARNING_RATE=0.001, EPOCHS=EPOCHS, head_size=16, ff_dim=4, num_transformer_blocks=2)
            model_INCEPTIONTIME,model_callbacks=model_INCEPTIONTIME(X_1,X_2,LEARNING_RATE=0.0001,EPOCHS=EPOCHS,BATCH_SIZE=2,stride=1, activation='linear',
                             nb_filters=30, use_residual=True, depth=8, kernel_size=41,use_bottleneck=False,
                             bottleneck_size = 32)
            
            #LSTM
            startLSTM=time.time()
            history_LSTM = model_LSTM.fit(X_train, y_train, batch_size=8, epochs=EPOCHS, verbose=1, shuffle=False,
                                callbacks=model_callbacks, validation_data=(X_test, y_test))
            trainLSTM=time.time()-startLSTM
            startLSTM=time.time()
            y_test_predict = model_LSTM.predict(X_test)
            testLSTM=time.time()-startLSTM
            y_test_predict_LSTM = scaler_y.inverse_transform(y_test_predict.reshape(-1, y_test_predict.shape[-1])).reshape(y_test_predict.shape) 
            y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)
            
            model_path = os.path.join(folder_path_perfold, "LSTM")
            os.makedirs(model_path, exist_ok=True)
            
            #SAVE MODEL
            save_model(model_LSTM, os.path.join(model_path, "model"))
    
            #SAVE MODEL SUMMARY
            summary_file_path = os.path.join(model_path,"model_summary.txt")
            with open(summary_file_path, "w") as f:
                with redirect_stdout(f):
                    model_LSTM.summary()
            #SAVE TRAIN/TEST TIME        
            time_path = os.path.join(model_path, "time.txt")
            with open(time_path, "w") as f:
                f.write(f"Training time is {trainLSTM} s ({trainLSTM/60} min)\n")
                f.write(f"Testing time is {testLSTM} s ({testLSTM/60} min)\n")
    
            #SAVE TEST PREDICTIONS
            pred_path = os.path.join(model_path, "test_predictions")
            os.makedirs(pred_path, exist_ok=True)
            testMAE_LSTM=[]
            testSMAPE_LSTM=[]
            
            for i in range(y_test.shape[0]):
                file_path_temp = os.path.join(pred_path, f"test_predictions_{i}.txt")
                
                with open(file_path_temp, 'w') as file:
                    j=0
                    for truth, prediction in zip(y_test_rescaled[i], y_test_predict_LSTM[i]):
                        
                        file.write("Truth: {}, prediction: {} \n".format(truth,prediction))
                        j+=1  
            
                indices = np.where(y_test_rescaled[i] == -1)[0]
                if len(indices) == 0:
                    indices=int(np.ceil(7573/REDUCE_DIM))
                else:
                    indices=indices[0]
               
                testMAE_LSTM.append(mean_absolute_error(y_test_rescaled[i,0:indices], y_test_predict_LSTM[i,0:indices]))
                testSMAPE_LSTM.append(smape(y_test_rescaled[i,0:indices], y_test_predict_LSTM[i,0:indices]))
            
            #SAVE TEST METRICS
            mae_path = os.path.join(model_path, 'mae_per_procedure.txt')
            with open(mae_path, 'w') as file:
                 file.write("Average among all the procedures in this fold:  {} [min]\n".format(np.mean(testMAE_LSTM)/60))
    
                 for index, item in enumerate(testMAE_LSTM):
                     file.write("Procedure number {} - mae: {} [min]\n".format(index,
                                item/60))
            smape_path = os.path.join(model_path, 'SMAPE_per_procedure.txt')
            with open(smape_path, 'w') as file:
                 file.write("Average among the procedures in this fold:  {} % %\n".format(np.mean(testSMAPE_LSTM)))
    
                 for index, item in enumerate(testSMAPE_LSTM):
                     file.write("Procedure number {} - SMAPE: {} % %\n".format(index,
                                  item))    
            
            
            #LSTM_FCN
            startLSTM_FCN=time.time()
            history_LSTM_FCN = model_LSTM_FCN.fit(X_train, y_train, batch_size=4, epochs=EPOCHS, verbose=1, shuffle=False,
                                callbacks=model_callbacks, validation_data=(X_test, y_test))
            trainLSTM_FCN=time.time()-startLSTM_FCN
            startLSTM_FCN=time.time()
            y_test_predict = model_LSTM_FCN.predict(X_test)
            testLSTM_FCN=time.time()-startLSTM_FCN
            y_test_predict_LSTM_FCN = scaler_y.inverse_transform(y_test_predict.reshape(-1, y_test_predict.shape[-1])).reshape(y_test_predict.shape) 
            y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)
            
            model_path = os.path.join(folder_path_perfold, "LSTM_FCN")
            os.makedirs(model_path, exist_ok=True)
            
            #SAVE MODEL
            save_model(model_LSTM_FCN, os.path.join(model_path, "model"))
    
            #SAVE MODEL SUMMARY
            summary_file_path = os.path.join(model_path,"model_summary.txt")
            with open(summary_file_path, "w") as f:
                with redirect_stdout(f):
                    model_LSTM_FCN.summary()
            #SAVE TRAIN/TEST TIME        
            time_path = os.path.join(model_path, "time.txt")
            with open(time_path, "w") as f:
                f.write(f"Training time is {trainLSTM_FCN} s ({trainLSTM_FCN/60} min)\n")
                f.write(f"Testing time is {testLSTM_FCN} s ({testLSTM_FCN/60} min)\n")
    
            #SAVE TEST PREDICTIONS
            pred_path = os.path.join(model_path, "test_predictions")
            os.makedirs(pred_path, exist_ok=True)
            testMAE_LSTM_FCN=[]
            testSMAPE_LSTM_FCN=[]
            
            for i in range(y_test.shape[0]):
                file_path_temp = os.path.join(pred_path, f"test_predictions_{i}.txt")
                
                with open(file_path_temp, 'w') as file:
                    j=0
                    for truth, prediction in zip(y_test_rescaled[i], y_test_predict_LSTM_FCN[i]):
                        
                        file.write("Truth: {}, prediction: {} \n".format(truth,prediction))
                        j+=1  
                indices = np.where(y_test_rescaled[i] == -1)[0]
                
                if len(indices) == 0:
                    indices=int(np.ceil(7573/REDUCE_DIM))
                else:
                    indices=indices[0]
               
                testMAE_LSTM_FCN.append(mean_absolute_error(y_test_rescaled[i,0:indices], y_test_predict_LSTM_FCN[i,0:indices]))
                testSMAPE_LSTM_FCN.append(smape(y_test_rescaled[i,0:indices], y_test_predict_LSTM_FCN[i,0:indices]))
            
            #SAVE TEST METRICS
            mae_path = os.path.join(model_path, 'mae_per_procedure.txt')
            with open(mae_path, 'w') as file:
                 file.write("Average among all the procedures in this fold:  {} [min]\n".format(np.mean(testMAE_LSTM_FCN)/60))
    
                 for index, item in enumerate(testMAE_LSTM_FCN):
                     file.write("Procedure number {} - mae: {} [min]\n".format(index,
                                item/60))
            smape_path = os.path.join(model_path, 'SMAPE_per_procedure.txt')
            with open(smape_path, 'w') as file:
                 file.write("Average among the procedures in this fold:  {} % %\n".format(np.mean(testSMAPE_LSTM_FCN)))
    
                 for index, item in enumerate(testSMAPE_LSTM_FCN):
                     file.write("Procedure number {} - SMAPE: {} % %\n".format(index,
                                  item))    
            
            #LSTM_Attention_ATTENTION
            startLSTM_Attention=time.time()
            history_LSTM_Attention = model_LSTM_Attention.fit(X_train, y_train, batch_size=8, epochs=EPOCHS, verbose=1, shuffle=False,
                                callbacks=model_callbacks, validation_data=(X_test, y_test))
            trainLSTM_Attention=time.time()-startLSTM_Attention
            startLSTM_Attention=time.time()
            y_test_predict = model_LSTM_Attention.predict(X_test)
            testLSTM_Attention=time.time()-startLSTM_Attention
            y_test_predict_LSTM_Attention = scaler_y.inverse_transform(y_test_predict.reshape(-1, y_test_predict.shape[-1])).reshape(y_test_predict.shape) 
            y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)
            
            model_path = os.path.join(folder_path_perfold, "LSTM_Attention")
            os.makedirs(model_path, exist_ok=True)
            
            #SAVE MODEL
            save_model(model_LSTM_Attention, os.path.join(model_path, "model"))
    
            #SAVE MODEL SUMMARY
            summary_file_path = os.path.join(model_path,"model_summary.txt")
            with open(summary_file_path, "w") as f:
                with redirect_stdout(f):
                    model_LSTM_Attention.summary()
            #SAVE TRAIN/TEST TIME        
            time_path = os.path.join(model_path, "time.txt")
            with open(time_path, "w") as f:
                f.write(f"Training time is {trainLSTM_Attention} s ({trainLSTM_Attention/60} min)\n")
                f.write(f"Testing time is {testLSTM_Attention} s ({testLSTM_Attention/60} min)\n")
    
            #SAVE TEST PREDICTIONS
            pred_path = os.path.join(model_path, "test_predictions")
            os.makedirs(pred_path, exist_ok=True)
            testMAE_LSTM_Attention=[]
            testSMAPE_LSTM_Attention=[]
            
            for i in range(y_test.shape[0]):
                file_path_temp = os.path.join(pred_path, f"test_predictions_{i}.txt")
                
                with open(file_path_temp, 'w') as file:
                    j=0
                    for truth, prediction in zip(y_test_rescaled[i], y_test_predict_LSTM_Attention[i]):
                        
                        file.write("Truth: {}, prediction: {} \n".format(truth,prediction))
                        j+=1  
            
                indices = np.where(y_test_rescaled[i] == -1)[0]
                if len(indices) == 0:
                    indices=int(np.ceil(7573/REDUCE_DIM))
                else:
                    indices=indices[0]
               
                testMAE_LSTM_Attention.append(mean_absolute_error(y_test_rescaled[i,0:indices], y_test_predict_LSTM_Attention[i,0:indices]))
                testSMAPE_LSTM_Attention.append(smape(y_test_rescaled[i,0:indices], y_test_predict_LSTM_Attention[i,0:indices]))
            
            #SAVE TEST METRICS
            mae_path = os.path.join(model_path, 'mae_per_procedure.txt')
            with open(mae_path, 'w') as file:
                 file.write("Average among all the procedures in this fold:  {} [min]\n".format(np.mean(testMAE_LSTM_Attention)/60))
    
                 for index, item in enumerate(testMAE_LSTM_Attention):
                     file.write("Procedure number {} - mae: {} [min]\n".format(index,
                                item/60))
            smape_path = os.path.join(model_path, 'SMAPE_per_procedure.txt')
            with open(smape_path, 'w') as file:
                 file.write("Average among the procedures in this fold:  {} % %\n".format(np.mean(testSMAPE_LSTM_Attention)))
    
                 for index, item in enumerate(testSMAPE_LSTM_Attention):
                     file.write("Procedure number {} - SMAPE: {} % %\n".format(index,
                                  item))    
            
            #INCEPTIONTIME
            startINCEPTIONTIME=time.time()
            history_INCEPTIONTIME = model_INCEPTIONTIME.fit(X_train, y_train, batch_size=2, epochs=EPOCHS, verbose=1, shuffle=False,
                                callbacks=model_callbacks, validation_data=(X_test, y_test))
            trainINCEPTIONTIME=time.time()-startINCEPTIONTIME
            startINCEPTIONTIME=time.time()
            y_test_predict = model_INCEPTIONTIME.predict(X_test)
            testINCEPTIONTIME=time.time()-startINCEPTIONTIME
            y_test_predict_INCEPTIONTIME = scaler_y.inverse_transform(y_test_predict.reshape(-1, y_test_predict.shape[-1])).reshape(y_test_predict.shape) 
            y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)
            
            model_path = os.path.join(folder_path_perfold, "INCEPTIONTIME")
            os.makedirs(model_path, exist_ok=True)
            
            #SAVE MODEL
            save_model(model_INCEPTIONTIME, os.path.join(model_path, "model"))
    
            #SAVE MODEL SUMMARY
            summary_file_path = os.path.join(model_path,"model_summary.txt")
            with open(summary_file_path, "w") as f:
                with redirect_stdout(f):
                    model_INCEPTIONTIME.summary()
            #SAVE TRAIN/TEST TIME        
            time_path = os.path.join(model_path, "time.txt")
            with open(time_path, "w") as f:
                f.write(f"Training time is {trainINCEPTIONTIME} s ({trainINCEPTIONTIME/60} min)\n")
                f.write(f"Testing time is {testINCEPTIONTIME} s ({testINCEPTIONTIME/60} min)\n")
    
            #SAVE TEST PREDICTIONS
            pred_path = os.path.join(model_path, "test_predictions")
            os.makedirs(pred_path, exist_ok=True)
            testMAE_INCEPTIONTIME=[]
            testSMAPE_INCEPTIONTIME=[]
            
            for i in range(y_test.shape[0]):
                file_path_temp = os.path.join(pred_path, f"test_predictions_{i}.txt")
                
                with open(file_path_temp, 'w') as file:
                    j=0
                    for truth, prediction in zip(y_test_rescaled[i], y_test_predict_INCEPTIONTIME[i]):
                        
                        file.write("Truth: {}, prediction: {} \n".format(truth,prediction))
                        j+=1  
            
                indices = np.where(y_test_rescaled[i] == -1)[0]
                if len(indices) == 0:
                    indices=int(np.ceil(7573/REDUCE_DIM))
                else:
                    indices=indices[0]
               
                testMAE_INCEPTIONTIME.append(mean_absolute_error(y_test_rescaled[i,0:indices], y_test_predict_INCEPTIONTIME[i,0:indices]))
                testSMAPE_INCEPTIONTIME.append(smape(y_test_rescaled[i,0:indices], y_test_predict_INCEPTIONTIME[i,0:indices]))
            
            #SAVE TEST METRICS
            mae_path = os.path.join(model_path, 'mae_per_procedure.txt')
            with open(mae_path, 'w') as file:
                 file.write("Average among all the procedures in this fold:  {} [min]\n".format(np.mean(testMAE_INCEPTIONTIME)/60))
    
                 for index, item in enumerate(testMAE_INCEPTIONTIME):
                     file.write("Procedure number {} - mae: {} [min]\n".format(index,
                                item/60))
            smape_path = os.path.join(model_path, 'SMAPE_per_procedure.txt')
            with open(smape_path, 'w') as file:
                 file.write("Average among the procedures in this fold:  {} % %\n".format(np.mean(testSMAPE_INCEPTIONTIME)))
    
                 for index, item in enumerate(testSMAPE_INCEPTIONTIME):
                     file.write("Procedure number {} - SMAPE: {} % %\n".format(index,
                                  item))    
            
            
            #transformer
            starttransformer=time.time()
            history_transformer = model_transformer.fit(X_train, y_train, batch_size=2, epochs=EPOCHS, verbose=1, shuffle=False,
                                callbacks=model_callbacks, validation_data=(X_test, y_test))
            traintransformer=time.time()-starttransformer
            starttransformer=time.time()
            y_test_predict = model_transformer.predict(X_test)
            testtransformer=time.time()-starttransformer
            y_test_predict_transformer = scaler_y.inverse_transform(y_test_predict.reshape(-1, y_test_predict.shape[-1])).reshape(y_test_predict.shape) 
            y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)
            
            model_path = os.path.join(folder_path_perfold, "transformer")
            os.makedirs(model_path, exist_ok=True)
            
            #SAVE MODEL
            save_model(model_transformer, os.path.join(model_path, "model"))
    
            #SAVE MODEL SUMMARY
            summary_file_path = os.path.join(model_path,"model_summary.txt")
            with open(summary_file_path, "w") as f:
                with redirect_stdout(f):
                    model_transformer.summary()
            #SAVE TRAIN/TEST TIME        
            time_path = os.path.join(model_path, "time.txt")
            with open(time_path, "w") as f:
                f.write(f"Training time is {traintransformer} s ({traintransformer/60} min)\n")
                f.write(f"Testing time is {testtransformer} s ({testtransformer/60} min)\n")
    
            #SAVE TEST PREDICTIONS
            pred_path = os.path.join(model_path, "test_predictions")
            os.makedirs(pred_path, exist_ok=True)
            testMAE_transformer=[]
            testSMAPE_transformer=[]
            
            for i in range(y_test.shape[0]):
                file_path_temp = os.path.join(pred_path, f"test_predictions_{i}.txt")
                
                with open(file_path_temp, 'w') as file:
                    j=0
                    for truth, prediction in zip(y_test_rescaled[i], y_test_predict_transformer[i]):
                        
                        file.write("Truth: {}, prediction: {} \n".format(truth,prediction))
                        j+=1  
            
                indices = np.where(y_test_rescaled[i] == -1)[0]
                if len(indices) == 0:
                    indices=int(np.ceil(7573/REDUCE_DIM))
                else:
                    indices=indices[0]
               
                testMAE_transformer.append(mean_absolute_error(y_test_rescaled[i,0:indices], y_test_predict_transformer[i,0:indices]))
                testSMAPE_transformer.append(smape(y_test_rescaled[i,0:indices], y_test_predict_transformer[i,0:indices]))
            
            #SAVE TEST METRICS
            mae_path = os.path.join(model_path, 'mae_per_procedure.txt')
            with open(mae_path, 'w') as file:
                 file.write("Average among all the procedures in this fold:  {} [min]\n".format(np.mean(testMAE_transformer)/60))
    
                 for index, item in enumerate(testMAE_transformer):
                     file.write("Procedure number {} - mae: {} [min]\n".format(index,
                                item/60))
            smape_path = os.path.join(model_path, 'SMAPE_per_procedure.txt')
            with open(smape_path, 'w') as file:
                 file.write("Average among the procedures in this fold:  {} % %\n".format(np.mean(testSMAPE_transformer)))
    
                 for index, item in enumerate(testSMAPE_transformer):
                     file.write("Procedure number {} - SMAPE: {} % %\n".format(index,
                                  item))    
            
            
            
            ###ENSEMBLE
            model_path = os.path.join(folder_path_perfold, "Ensemble")
            os.makedirs(model_path, exist_ok=True)
            
            
            y_test_predict_ensemble=0.2*y_test_predict_INCEPTIONTIME+0.2*y_test_predict_LSTM+0.2*y_test_predict_LSTM_Attention+0.2*y_test_predict_LSTM_FCN+0.2*y_test_predict_transformer
            #SAVE MODEL
      #      save_model(model_transformer, os.path.join(model_path, "model"))
    
            #SAVE MODEL SUMMARY
        #    summary_file_path = os.path.join(model_path,"model_summary.txt")
         #   with open(summary_file_path, "w") as f:
          #      with redirect_stdout(f):
           #         model.summary()
            #SAVE TRAIN/TEST TIME        
           # time_path = os.path.join(model_path, "time.txt")
           # with open(time_path, "w") as f:
           #     f.write(f"Training time is {traintransformer} s ({traintransformer}/60 min)\n")
           #     f.write(f"Testing time is {testtransformer} s ({testtransformer}/60 min)\n")
    
            #SAVE TEST PREDICTIONS
            pred_path = os.path.join(model_path, "test_predictions")
            os.makedirs(pred_path, exist_ok=True)
            testMAE_transformer=[]
            testSMAPE_transformer=[]
            
            for i in range(y_test.shape[0]):
                file_path_temp = os.path.join(pred_path, f"test_predictions_{i}.txt")
                
                with open(file_path_temp, 'w') as file:
                    j=0
                    for truth, prediction in zip(y_test_rescaled[i], y_test_predict_ensemble[i]):
                        
                        file.write("Truth: {}, prediction: {} \n".format(truth,prediction))
                        j+=1  
            
                indices = np.where(y_test_rescaled[i] == -1)[0]
                if len(indices) == 0:
                    indices=int(np.ceil(7573/REDUCE_DIM))
                else:
                    indices=indices[0]
               
                testMAE_transformer.append(mean_absolute_error(y_test_rescaled[i,0:indices], y_test_predict_ensemble[i,0:indices]))
                testSMAPE_transformer.append(smape(y_test_rescaled[i,0:indices], y_test_predict_ensemble[i,0:indices]))
            
            #SAVE TEST METRICS
            mae_path = os.path.join(model_path, 'mae_per_procedure.txt')
            with open(mae_path, 'w') as file:
                 file.write("Average among all the procedures in this fold:  {} [min]\n".format(np.mean(testMAE_transformer)/60))
    
                 for index, item in enumerate(testMAE_transformer):
                     file.write("Procedure number {} - mae: {} [min]\n".format(index,
                                item/60))
            smape_path = os.path.join(model_path, 'SMAPE_per_procedure.txt')
            with open(smape_path, 'w') as file:
                 file.write("Average among the procedures in this fold:  {} % %\n".format(np.mean(testSMAPE_transformer)))
    
                 for index, item in enumerate(testSMAPE_transformer):
                     file.write("Procedure number {} - SMAPE: {} % %\n".format(index,
                                  item))    
            
            
            
            
            
            


        
        
        
        
 


        
            # LSTM
            # LSTM-FCN
            # Attention-LSTM
            # InceptionTime
            # Transformer
            # Ensemble
            
            #Keep track of:
                #Training time
                #Testing time
                #Test predictions vs truth
                #Error at different time stamps of the model
                #Save model
                #Save model info (num parameters,...)
    