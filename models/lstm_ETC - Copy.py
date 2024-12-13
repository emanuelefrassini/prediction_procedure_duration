import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from keras.layers import LSTM, Dropout, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import time
import os
import datetime
#PARAMETERS
pc='PC_office'
endpoint='time_binary_regression'
#MODEL PARAMETERS

EPOCHS=1000
NUM_FOLDS=3

VERBOSE=1
BATCH_SIZE=8
REDUCE_DIM=5
UNITS=128
DROPOUT_RATE=0.1
LEARNING_RATE=0.001
#FOLDER CREATION

if pc=='PC_mine' or pc=='PC_office':
    base_folder = r'C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Codes\MTSC\runs\\'
elif pc=='PC_tudelft' or pc=='PC_rdg':
    base_folder = r'C:\Users\efrassini\OneDrive - Delft University of Technology\Work\Codes\MTSC\runs\\'

# Get the current date and time
current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Create the folder name with the current date and time
folder_name = f'runs_{current_datetime}_{endpoint}'

# Path for the new folder
folder_path_inforun = os.path.join(base_folder, folder_name)

# Info run save path
os.makedirs(folder_path_inforun, exist_ok=True)


# Model save path
folder_path_save_model = os.path.join(folder_path_inforun, 'model')
os.makedirs(folder_path_save_model, exist_ok=True)

# Training plots save path
folder_path_training = os.path.join(folder_path_inforun, 'train')
os.makedirs(folder_path_training, exist_ok=True)

# Test plots save path
folder_path_test = os.path.join(folder_path_inforun, 'test')
os.makedirs(folder_path_test, exist_ok=True)

# Define the log directory for TensorBoard
folder_path_tensorboard = os.path.join(folder_path_inforun, 'log_tensorboard')
os.makedirs(folder_path_tensorboard, exist_ok=True)

#log_dir = folder_path_tensorboard +'\\'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=folder_path_tensorboard, histogram_freq=1)


current_dir = os.getcwd()


# Construct the path to the dataset relative to the script
X_path = os.path.join(current_dir, "..\data\X.csv")
Y_path = os.path.join(current_dir, "..\data\Y.csv")

X = np.loadtxt(X_path,delimiter=",").reshape(222, 7573, 1)
Y = np.loadtxt(Y_path,delimiter=",").reshape(222, 7573, 1)

#loaded_Y = np.loadtxt("data\Y.csv", delimiter=",").reshape(222, 7573, 1)
# Verify if they are identical
 




import tensorflow.keras.backend as K

def calculate_corridor_func(video_length, mean_video_length, d=5):
    """
    Calculates the corridor function, presented in the paper:
    https://arxiv.org/pdf/2002.11367.pdf. 
    This implementation follows the same notions as the paper.
    """
    c_x = tf.range(video_length)

    g_t = video_length - c_x
    n_t = tf.maximum(mean_video_length - c_x, tf.zeros_like(c_x))

    a_t = 1 - (2 / (1 + tf.exp((c_x / video_length) * d)))

    c_t = (a_t * g_t) + ((1 - a_t) * n_t)

    return c_t


def calculate_corridor_mask(preds, labels, mean_video_length, d=5, tolerance=0):
    """
    Calculates mask of which pred lays between the corridor function and label.
    Following https://arxiv.org/pdf/2002.11367.pdf.
    This implementation follows the same notions as the paper
    """
    c_t = calculate_corridor_func(
        video_length=tf.shape(preds)[0],
        mean_video_length=mean_video_length,
        d=d,
    )

    mask = tf.logical_or(
        tf.logical_and(c_t <= preds, preds <= labels + tolerance),
        tf.logical_and(labels - tolerance <= preds, preds <= c_t),
    )

    return mask, c_t


def calculate_corridor_weights(
    preds,
    labels,
    video_length,
    mean_video_length,
    d=5,
    tolerance=0,
    off_corridor_penalty=1,
    padding_value=-1
):
    """
    Calculates the loss weight for each index.
    Following https://arxiv.org/pdf/2002.11367.pdf.
    This implementation follows the same notions as the paper.
    """
    w = tf.ones(tf.shape(preds)[0] - video_length) * padding_value

    p = preds[:video_length]
    l = labels[:video_length]
    mask, c_t = calculate_corridor_mask(
        preds=p, labels=l, mean_video_length=mean_video_length, d=d, tolerance=tolerance
    )

    weights = tf.pow(tf.abs(p - l) / tf.abs(c_t - l), 2)

    weights = tf.where(tf.math.logical_not(mask), off_corridor_penalty, weights)
    weights = tf.concat([weights, w], axis=0)
    print(tf.shape(weights))
    return weights




def smape_loss(y_true, y_pred):
    epsilon = 0.1
    summ = K.maximum(K.abs(y_true) + K.abs(y_pred) + epsilon, 0.5 + epsilon)
   # smape = K.abs(y_pred - y_true) / summ * 2.0
    smape = K.abs(y_pred - y_true) / summ 
    print(tf.shape(smape))
    return smape

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum( np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))







#%%LSTM TEST AND SAVE



if REDUCE_DIM:
    X=X[:,0::REDUCE_DIM,:]
    Y=Y[:,0::REDUCE_DIM,:]

def create_model():
    model = keras.Sequential()
    model.add(keras.layers.Masking(mask_value=-1))
    
    model.add((LSTM(UNITS, return_sequences=True, stateful=False)))
    #model.add((LSTM(UNITS, return_sequences=True, stateful=False)))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(1, activation='relu'))
    
    #model.compile(loss='mean_absolute_error',  # weighted_categorical_crossentropy(weights),
     #             optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE), metrics='mae')
    model.compile(loss=smape_loss,  # weighted_categorical_crossentropy(weights),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), metrics=smape_loss)
    
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=max(
        2, np.round(EPOCHS/10)), restore_best_weights=True, min_delta=0.001)
    lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=max(2, np.round(EPOCHS/20)))
    
    # self.model.stop_training = True
    # ,MyThresholdCallback(),MyThresholdCallback2()]
    model_callbacks = [early_stopping, lr_reduction,tensorboard_callback]
    return model,model_callbacks


smape_average=100
mae_average=100
while smape_average>17 and mae_average>7:
    accuracy_per_slice = []
    std_accuracy_per_slice = []
    time_per_slice = []
    
        # Experiment set up
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True)
    loss_train_per_fold = []
    accuracy_train_per_fold = []
    std_train_per_fold = []
    
    loss_test_per_fold = []
    accuracy_test_per_fold = []
    std_test_per_fold = []
    
    time_fold = []
    
    fold = 0
    cm_list = []
    accuracy_per_class_list = []
    accuracy_SMAPE_per_fold=[]
    accuracy_baseline_mae_per_fold=[]
    accuracy_baseline_smape_per_fold=[]
    
    for train, test in kfold.split(X):
        start_fold = time.time()
    
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
      
        model,model_callbacks=create_model()
    
        history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, shuffle=False,
                            callbacks=model_callbacks, validation_data=(X_test, y_test))
    
        print("\nFold {} took {} seconds".format(
            fold, time.time()-start_fold))
    
        print("\n---------------------------------------------------------")
    
        y_train_predict = model.predict(X_train)
        y_train_predict = scaler_y.inverse_transform(y_train_predict.reshape(-1, y_train_predict.shape[-1])).reshape(y_train_predict.shape) 
        y_train = scaler_y.inverse_transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train.shape)
        
        
        
        
        y_test_predict = model.predict(X_test)
        y_test_predict = scaler_y.inverse_transform(y_test_predict.reshape(-1, y_test_predict.shape[-1])).reshape(y_test_predict.shape) 
        y_test = scaler_y.inverse_transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)
        
    
        
        # calculate root mean squared error
        trainScore_tot=[]
        trainSMAPE_tot=[]
    
        for i in range(y_train.shape[0]):
            trainScore_tot.append(mean_absolute_error(y_train[i], y_train_predict[i]))
            trainSMAPE_tot.append(smape(y_train[i], y_train_predict[i]))
    
    
        testScore_tot=[]
        testSMAPE_tot=[]
        
        baseline_mae=[]
        baseline_smape=[]
        
        baseline=(np.zeros(((int(np.ceil(7573/REDUCE_DIM))),1)))
        for h in range(baseline.shape[0]):
            baseline[h,0]=-1
            if h<int(np.ceil(2700/REDUCE_DIM)):
                baseline[h,0]=2700-h*REDUCE_DIM
    
       
        for i in range(y_test.shape[0]):
             indices = np.where(y_test[i] == -1)[0]
             if len(indices) == 0:
                 indices=int(np.ceil(7573/REDUCE_DIM))
             else:
                 indices=indices[0]
            
             testScore_tot.append(mean_absolute_error(y_test[i,0:indices], y_test_predict[i,0:indices]))
             testSMAPE_tot.append(smape(y_test[i,0:indices], y_test_predict[i,0:indices]))
           #  baseline_mae.append(mean_absolute_error(y_test[i,0:indices],baseline[0:indices]))
           #  baseline_smape.append(smape(y_test[i,0:indices], baseline[0:indices]))
    
    
        if not REDUCE_DIM:
            trainScore_10=[]
            y_train_10=y_train[:,0:10*60]
            y_train_predict_10=y_train_predict[:,0:10*60]
            for i in range(y_train_10.shape[0]):
                trainScore_10.append(mean_absolute_error(y_train_10[i], y_train_predict_10[i]))
           
            trainScore_20=[]
            y_train_20=y_train[:,10*60:20*60]
            y_train_predict_20=y_train_predict[:,10*60:20*60]
            for i in range(y_train_20.shape[0]):
                trainScore_20.append(mean_absolute_error(y_train_20[i], y_train_predict_20[i]))
           
            trainScore_30=[]
            y_train_30=y_train[:,20*60:30*60]
            y_train_predict_30=y_train_predict[:,20*60:30*60]
            for i in range(y_train_30.shape[0]):
                trainScore_30.append(mean_absolute_error(y_train_30[i], y_train_predict_30[i]))
           
            trainScore_40=[]
            y_train_40=y_train[:,30*60:40*60]
            y_train_predict_40=y_train_predict[:,30*60:40*60]
            for i in range(y_train_40.shape[0]):
                trainScore_40.append(mean_absolute_error(y_train_40[i], y_train_predict_40[i]))
        
    
    
            
            testScore_10=[]
            y_test_10=y_test[:,0:10*60]
            y_test_predict_10=y_test_predict[:,0:10*60]
            for i in range(y_test_10.shape[0]):
                testScore_10.append(mean_absolute_error(y_test_10[i], y_test_predict_10[i]))
           
            testScore_20=[]
            y_test_20=y_test[:,10*60:20*60]
            y_test_predict_20=y_test_predict[:,10*60:20*60]
            for i in range(y_test_20.shape[0]):
                testScore_20.append(mean_absolute_error(y_test_20[i], y_test_predict_20[i]))
           
            testScore_30=[]
            y_test_30=y_test[:,20*60:30*60]
            y_test_predict_30=y_test_predict[:,20*60:30*60]
            for i in range(y_test_30.shape[0]):
                testScore_30.append(mean_absolute_error(y_test_30[i], y_test_predict_30[i]))
           
            testScore_40=[]
            y_test_40=y_test[:,30*60:40*60]
            y_test_predict_40=y_test_predict[:,30*60:40*60]
            for i in range(y_test_40.shape[0]):
                testScore_40.append(mean_absolute_error(y_test_40[i], y_test_predict_40[i]))
    
    
    
    
    
    
    
    
    
    
    
    
    
        accuracy_train_per_fold.append(np.mean(trainScore_tot))
        std_train_per_fold.append(np.std(trainScore_tot))
    
       # accuracy_train_per_fold.append(np.mean(history.history['categorical_accuracy']))
       # print("\n-----------------------------------------------------")
       # print("\nTest Set Evaluation - Loss: {:.4f}, Accuracy: {:.4f}".format(evaluation[0], evaluation[1]))
       # loss_test_per_fold.append(evaluation[0])
        accuracy_test_per_fold.append(np.mean(testScore_tot))
        std_test_per_fold.append(np.std(testScore_tot))
        
        accuracy_SMAPE_per_fold.append(np.mean(testSMAPE_tot))
        
        accuracy_baseline_mae_per_fold.append(np.mean(baseline_mae))
        accuracy_baseline_smape_per_fold.append(np.mean(baseline_smape))
    
        time_fold.append(time.time()-start_fold)
    
    
    smape_average=np.mean(accuracy_SMAPE_per_fold)
    mae_average=np.mean(accuracy_test_per_fold)/60
    
file_path2 = os.path.join(folder_path_inforun, "test_predictions")
os.makedirs(file_path2, exist_ok=True)

for i in range(y_test.shape[0]):
    file_path_temp = os.path.join(file_path2, f"test_predictions_{i}.txt")
    
    with open(file_path_temp, 'w') as file:
        j=0
        for truth, prediction in zip(y_test[i], y_test_predict[i]):
            
            file.write("Truth: {}, prediction: {} \n".format(truth,prediction))
            j+=1    
##file_path2 = os.path.join(folder_path_test, "test_predictions")
#os.makedirs(file_path2, exist_ok=True)

        
##     np.mean(accuracy_test_per_fold),))
#accuracy_per_slice.append(np.mean(accuracy_test_per_fold))
#std_accuracy_per_slice.append(np.std(accuracy_test_per_fold))
#time_per_slice.append(time.time()-start_slice)
        # keras.backend.clear_session()
        # Create x-axis values (assuming indices as x-axis)
    # Generate ticks: 5, 10, 15, 20, ...
custom_ticks = [5 * (i+1) for i in range(len(accuracy_per_slice))]

x_values = custom_ticks
 # x_values = range(len(accuracy_per_slice))
testplot=[test/60 for test in testScore_tot]
 # Plotting the list
plt.plot(range(len(testScore_tot)), testplot, marker='o', linestyle='-')
plt.xlabel('Elapsed time')
plt.ylabel('mae [s]')
plt.title('Error in the prediction of the duration of the procedure')

plot_file_path = os.path.join(folder_path_inforun, 'mae_per_procedure.png')

plt.savefig(plot_file_path)  # Save the heatmap to the specified file
plt.close()

if VERBOSE:
    file_path2 = os.path.join(folder_path_inforun, "ETC_per_procedure")
    os.makedirs(file_path2, exist_ok=True)
    for i in range(y_test.shape[0]):
        file_path_temp = os.path.join(file_path2, f"ETC_procedure_{i}.png")
        #x_values = range(int(y_test[i, 0, 0]/60), -REDUCE_DIM)
      #  x_values = range(0,int(y_test[i, 0, 0]), REDUCE_DIM)
        x_values = [value / 60 for value in range(0,int(y_test[i, 0, 0]), REDUCE_DIM)]

        plt.figure()
        plt.plot(x_values, y_test[i,:len(x_values),:]/60, 'bo',label='Real ETC')
        plt.plot(x_values, y_test_predict[i,:len(x_values),:]/60, 'ro',label='Predicted ETC')

    
        plt.xlabel('Elapsed time [min] ')
        plt.ylabel('ETC [min]')
        plt.title(f'ETC for procedure {i}')
        
        plt.legend()
        plt.savefig(file_path_temp)  # Save the heatmap to the specified file
        plt.close()


# plt.figure(figsize=(10, 6))
 # bars = plt.bar(classes, phase_counts, width=0.4)

 # plt.xlabel('Class')
 # plt.ylabel('Abundance')
 # plt.title('Abundance Distribution of {} Classes'.format(NUM_CLASSES))

 # Adding annotations on top of the bars
 # for bar, count in zip(bars, phase_counts):
 #   plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), np.round(count,3),
 #           ha='center', va='bottom', fontsize=8)

 # plot_file_path = os.path.join(folder_path_inforun, 'abbundance_per_class.png')

# plt.savefig(plot_file_path)  # Save the heatmap to the specified file
 # plt.close()

file_path = os.path.join(folder_path_inforun, 'time.txt')
with open(file_path, 'w') as file:
     for item in time_per_slice:
         file.write("%s\n" % item)

file_path = os.path.join(folder_path_inforun, 'mae_per_procedure.txt')
with open(file_path, 'w') as file:
     file.write("Average among all the procedures:  {} [min]\n".format(np.mean(accuracy_test_per_fold)/60))

     for index, item in enumerate(testScore_tot):
         file.write("Procedure number {} - mae: {} [min]\n".format(index,
                    item/60))
file_path = os.path.join(folder_path_inforun, 'SMAPE_per_procedure.txt')
with open(file_path, 'w') as file:
     file.write("Average among all the procedures:  {} % %\n".format(np.mean(accuracy_SMAPE_per_fold)))

     for index, item in enumerate(testSMAPE_tot):
         file.write("Procedure number {} - SMAPE: {} % %\n".format(index,
                    item))



mae=[]

duration=[]
smape_list=[]
mae_interq=[]
duration_interq=[]
smape_interq=[]
for i in range(y_test.shape[0]):
    mae.append(mean_absolute_error(y_test[i],y_test_predict[i])/60)
    smape_list.append(smape(y_test[i],y_test_predict[i]))

    duration.append(((y_test[i,0])/60).astype(np.float32))
    if 10<duration[i]<72:
        mae_interq.append(mean_absolute_error(y_test[i],y_test_predict[i])/60)
        smape_interq.append(smape(y_test[i],y_test_predict[i]))

        duration_interq.append(((y_test[i,0])/60).astype(np.float32))

plt.figure(figsize=(8, 6))
plt.plot(mae, 'o',label='MAE')
plt.plot(duration,'x', label='Duration')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Comparison of MAE and Duration')
plt.legend()
plt.grid(True)

# Plot dashed vertical lines connecting the points
for i in range(len(mae)):
    plt.vlines(i, min(mae[i], duration[i]), max(mae[i], duration[i]), linestyle='--', color='gray')
plot_path=os.path.join(folder_path_test,'comparison_mae_duration')
plt.savefig(plot_path)
plt.close()


y_test_predict2=model.predict(X_test)
y_test_predict2 = scaler_y.inverse_transform(y_test_predict2.reshape(-1, y_test_predict2.shape[-1])).reshape(y_test_predict2.shape) 

q1=np.quantile(duration,0)
q2=np.quantile(duration,1)
y_test_copy = np.array([y_test[i] for i in range(y_test.shape[0]) if q1<duration[i]<q2])
y_test_copy_predict = np.array([y_test_predict[i] for i in range(y_test_predict.shape[0]) if q1<duration[i]<q2])






y_test_copy=y_test
y_test_copy_predict=y_test_predict


correct_5=np.zeros((y_test_copy.shape[0],1))
correct_10=np.zeros((y_test_copy.shape[0],1))
correct_15=np.zeros((y_test_copy.shape[0],1))
correct_20=np.zeros((y_test_copy.shape[0],1))
for i in range(y_test_copy.shape[0]):
    ind_5 = (np.abs(y_test_copy_predict[i] -60*5)).argmin()
  #  ind=np.where( abs(y_test_copy_predict[i]-60*howmany)<REDUCE_DIM)[0][0]
    if y_test_copy[i,ind_5]<60*5 and y_test_copy[i,ind_5]!=-1:
        correct_5[i]=1
    ind_10 = (np.abs(y_test_copy_predict[i] -60*10)).argmin()
    #  ind=np.where( abs(y_test_copy_predict[i]-60*howmany)<REDUCE_DIM)[0][0]
    if y_test_copy[i,ind_10]<60*10 and y_test_copy[i,ind_10]!=-1:
          correct_10[i]=1
    ind_15 = (np.abs(y_test_copy_predict[i] -60*15)).argmin()
     #  ind=np.where( abs(y_test_copy_predict[i]-60*howmany)<REDUCE_DIM)[0][0]
    if y_test_copy[i,ind_15]<60*15 and y_test_copy[i,ind_15]!=-1:
           correct_15[i]=1
    ind_20 = (np.abs(y_test_copy_predict[i] -60*20)).argmin()
    #  ind=np.where( abs(y_test_copy_predict[i]-60*howmany)<REDUCE_DIM)[0][0]
    if y_test_copy[i,ind_20]<60*20 and y_test_copy[i,ind_20]!=-1:
          correct_20[i]=1
           
def compute_accuracy(arr):
    count = 0
    length=arr.shape[0]
    for element in arr:
        if element == 1:
            count += 1
    return count/length

accuracy_5=compute_accuracy(correct_5)
accuracy_10=compute_accuracy(correct_10)
accuracy_15=compute_accuracy(correct_15)
accuracy_20=compute_accuracy(correct_20)

# Create the file path
file_path = folder_path_inforun + "/accuracy_values.txt"

# Write the values to the file
with open(file_path, 'w') as file:
    file.write(f"Accuracy_5: {accuracy_5}\n")
    file.write(f"Accuracy_10: {accuracy_10}\n")
    file.write(f"Accuracy_15: {accuracy_15}\n")
    file.write(f"Accuracy_20: {accuracy_20}\n")

file_path = folder_path_inforun + "/info_run.txt"

# Write the values to the file
with open(file_path, 'w') as file:
    file.write(f"Units: {UNITS}\n")
    file.write(f"Reduce dim: {REDUCE_DIM}\n")
    file.write(f"Dropout: {DROPOUT_RATE}\n")
    file.write(f"Learning rate: {LEARNING_RATE}\n")
    file.write(f"Batch size: {BATCH_SIZE}\n")


scaler = MinMaxScaler()

# Fit and transform y_true
#y_true_scaled = scaler.fit_transform(y_test)

# Transform y_pred using the same scaler
#y_pred_scaled = scaler.transform(y_test_predict)
    
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
plot_path_2=os.path.join(folder_path_test,'Scaled average ETC')
plt.savefig(plot_path_2)

print("\nEverything has been saved succesfully!")


r'''
error=5
correct_5=np.zeros((y_test_copy.shape[0],1))
correct_10=np.zeros((y_test_copy.shape[0],1))
correct_20=np.zeros((y_test_copy.shape[0],1))
correct_30=np.zeros((y_test_copy.shape[0],1))
correct_40=np.zeros((y_test_copy.shape[0],1))
for i in range(y_test_copy.shape[0]):
    if np.ceil(abs(y_test_copy[i,int(5*60/REDUCE_DIM)]-y_test_copy_predict[i,int(5*60/REDUCE_DIM)])/60)<error:
        correct_5[i]=1
    if np.ceil(abs(y_test_copy[i,int(10*60/REDUCE_DIM)]-y_test_copy_predict[i,int(10*60/REDUCE_DIM)])/60)<error:
        correct_10[i]=1
    if np.ceil(abs(y_test_copy[i,int(20*60/REDUCE_DIM)]-y_test_copy_predict[i,int(20*60/REDUCE_DIM)])/60)<error:
        correct_20[i]=1
    if np.ceil(abs(y_test_copy[i,int(30*60/REDUCE_DIM)]-y_test_copy_predict[i,int(30*60/REDUCE_DIM)])/60)<error:
        correct_30[i]=1     
    if np.ceil(abs(y_test_copy[i,int(40*60/REDUCE_DIM)]-y_test_copy_predict[i,int(40*60/REDUCE_DIM)])/60)<error:
        correct_40[i]=1

accuracy_5=compute_accuracy(correct_5)
accuracy_10=compute_accuracy(correct_10)
accuracy_20=compute_accuracy(correct_20)
accuracy_30=compute_accuracy(correct_30)
accuracy_40=compute_accuracy(correct_40)


import os
import re

# Define the folder path where the text files are located
folder_path = r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Codes\MTSC\runs\runs_20240407-112330_time_binary_regression\test_predictions"

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

'''

