from tensorboard.plugins.hparams import api as hp
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from keras.models import Model
from keras import layers
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

BATCH_SIZE=4
REDUCE_DIM=10
LEARNING_RATE=0.001
HEAD_SIZE=32
FFD=4
NUM_TRANS_BLOCKS=2

SMAPE_MIN=18.5
MAE_MIN=7.5
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






#%%#MODEL CREATION HP SEARCH
def train_test_model_custom(hparams,X,Y):
    try:
        REDUCE_DIM=hparams[HP_REDUCE_DIM]
        LEARNING_RATE=hparams[HP_LEARNINGRATE]
        BATCH_SIZE=hparams[HP_BATCHSIZE]
        
        num_transf_blocks=hparams[HP_NUM_TRANSFORMER_BLOCKS]
        head_size=hparams[HP_HEAD_SIZE]
        ffd=hparams[HP_FFD]
        if REDUCE_DIM:
            X=X[:,0::REDUCE_DIM,:]
            Y=Y[:,0::REDUCE_DIM,:]
        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
            # Attention and Normalization
            x = layers.MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout
            )(inputs, inputs)
            x = layers.Dropout(dropout)(x)
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            res = x + inputs
        
            # Feed Forward Part
            x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
            x = layers.Dropout(dropout)(x)
            x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            return x + res
        def create_model(
            head_size=head_size,
            num_heads=4,
            ff_dim=ffd,
            num_transformer_blocks=num_transf_blocks,
            mlp_units=[128],
            mlp_dropout=0.4,
            dropout=0.25,
                ):
            inputs = keras.Input(shape=(X.shape[1],X.shape[2]))
            x = inputs
            for _ in range(num_transformer_blocks):
                x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
        
            #x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
            for dim in mlp_units:
                x = layers.Dense(dim, activation="relu")(x)
                x = layers.Dropout(mlp_dropout)(x)
            outputs = layers.Dense(1, activation="relu")(x)
            model = Model(inputs, outputs)

            model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), metrics='mae')
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=max(
                2, np.round(EPOCHS/10)), restore_best_weights=True, min_delta=0.001)
            lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=max(2, np.round(EPOCHS/20)))
            
            # self.model.stop_training = True
            # ,MyThresholdCallback(),MyThresholdCallback2()]
            model_callbacks = [early_stopping, lr_reduction,tensorboard_callback]
            return model,model_callbacks

        
            # Experiment set up
        kfold = KFold(n_splits=NUM_FOLDS, shuffle=True)
        accuracy_train_per_fold = []
        std_train_per_fold = []
        
        accuracy_test_per_fold = []
        std_test_per_fold = []
        
        time_fold = []
        
        fold = 0
        accuracy_SMAPE_per_fold=[]
        
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
        
            model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, shuffle=False,
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
            
         #   accuracy_baseline_mae_per_fold.append(np.mean(baseline_mae))
          #  accuracy_baseline_smape_per_fold.append(np.mean(baseline_smape))
        
            time_fold.append(time.time()-start_fold)
            
        
        a=np.mean(accuracy_test_per_fold)/60
        b=np.mean(accuracy_SMAPE_per_fold)
        
    except ValueError:
        a=np.float64(100)
        b=np.float64(100)
    
    
    return a,b
    



HP_LEARNINGRATE= hp.HParam('learning_rate', hp.Discrete([0.1,0.01,0.001,0.0001]))
HP_BATCHSIZE=hp.HParam('batch_size', hp.Discrete([2,4,8,16,32]))#,16,32,64]))
HP_REDUCE_DIM=hp.HParam('reduce_dim', hp.Discrete([10]))#,30,45,60]))
HP_NUM_TRANSFORMER_BLOCKS= hp.HParam('num_transformer_blocks', hp.Discrete([2,4]))
HP_HEAD_SIZE= hp.HParam('head_size', hp.Discrete([4,8,16,32]))
HP_FFD= hp.HParam('ffd', hp.Discrete([2,4,8]))


METRIC_ACCURACY_mae = 'mae'
METRIC_ACCURACY_smape = 'smape'

hyperparam_folder=  os.path.join(folder_path_tensorboard, 'hparam_tuning/')
os.makedirs(hyperparam_folder, exist_ok=True)

with tf.summary.create_file_writer(hyperparam_folder).as_default():
  hp.hparams_config(
      hparams=[HP_REDUCE_DIM,HP_NUM_TRANSFORMER_BLOCKS, HP_HEAD_SIZE,HP_LEARNINGRATE,HP_FFD,HP_BATCHSIZE],
    metrics=[hp.Metric(METRIC_ACCURACY_mae, display_name='MAE'),
             hp.Metric(METRIC_ACCURACY_smape,display_name='SMAPE')],)
  
  
def run_costum(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy_mae,accuracy_smape = train_test_model_custom(hparams,X,Y)
    tf.summary.scalar(METRIC_ACCURACY_mae, accuracy_mae, step=1)
    tf.summary.scalar(METRIC_ACCURACY_smape, accuracy_smape, step=1)


 

session_num = 0
for reduce_dim in HP_REDUCE_DIM.domain.values:
    for batchsize in HP_BATCHSIZE.domain.values:
      for lr in HP_LEARNINGRATE.domain.values:
        for ffd in HP_FFD.domain.values:
            for numblocks in HP_NUM_TRANSFORMER_BLOCKS.domain.values:
                for headsize in HP_HEAD_SIZE.domain.values:
                  hparams = {
                      HP_REDUCE_DIM:reduce_dim,
                      HP_BATCHSIZE: batchsize,
                      HP_LEARNINGRATE: lr,
                      HP_FFD:ffd,
                      HP_NUM_TRANSFORMER_BLOCKS: numblocks,
                      HP_HEAD_SIZE: headsize
                  }
                  run_name = "run-%d" % session_num
                  print('--- Starting trial: %s' % run_name)
                  print({h.name: hparams[h] for h in hparams})
                  run_costum(hyperparam_folder + run_name, hparams)
                  session_num += 1



print("\nEverything has been saved succesfully!")