import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
to_categorical = tf.keras.utils.to_categorical

def prepare_data(pc,granularity,cutoff=[0,130],verbose=0,columns_to_drop=None,columns_to_keep=None,source=None,reduce_dim=None):
    if pc=='PC_mine':
        if granularity!='time_binary_regression' and granularity!='time_binary':
            folder_data = r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Codes\MTSC\data\data_numpy.pkl"
        elif granularity=='time_binary_regression' or granularity=='time_binary':
            folder_data = r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Codes\MTSC\data\data_numpy_with_time.pkl"
    elif pc=='PC_rdg' or pc=='PC_tudelft':
        if granularity!='time_binary_regression' and granularity!='time_binary':
            folder_data = r"C:\Users\efrassini\OneDrive - Delft University of Technology\Work\Codes\MTSC\data\data_numpy.pkl"
        elif granularity=='time_binary_regression' or granularity=='time_binary':
            folder_data = r"C:\Users\efrassini\OneDrive - Delft University of Technology\Work\Codes\MTSC\data\data_numpy_with_time.pkl"

    folder_data=r"/scratch/emanuelefrassi/transformers/data_numpy_with_time.pkl"
    with open(folder_data, 'rb') as file:
        loaded_data = pickle.load(file)
    
    Y_ini = loaded_data['target']

        
    r'''  
    if pc=='PC_rdg' or pc=='PC_tudelft':
        data_list=r"C:\Users\efrassini\OneDrive - Delft University of Technology\Work\Codes\MTSC\data\data_complete_list.pkl"
    elif pc=='PC_mine':
        data_list=r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Codes\MTSC\data\data_complete_list.pkl"
    '''
    data_list=r"/scratch/emanuelefrassi/transformers/data_complete_list.pkl"
    

    with open(data_list, 'rb') as file:
        loaded_data = pickle.load(file)
    X_phase_only=np.zeros((228,7573,2))
    
    for j in range(len(loaded_data)):
       X_phase_only[j,:,0]=loaded_data[j]["Phase_num"]
       X_phase_only[j,:,1]=loaded_data[j]["CumTime"]

       loaded_data[j].drop(columns="Phase_num",inplace=True)
    
    columns_drop=columns_to_drop
    columns_tokeep=[]
    if 'system log' in source:
        columns_tokeep.extend(['ShutterPositionX','ShutterPositionY','WedgeLeftDistance',
                        'WedgeLeftAngle','WedgeRightDistance','WedgeRightAngle',
                        'PositionCarm','PositionDetector','PositionPropellor',
                        'FrontalBeamLongitudinal','FrontalBeamTransversal',
                        'FrontalRotateDetector','FrontalSwing','FrontalZrotation',
                        'TableHeight','TableLateral','TableLongitudinal',
                        'AcqCount','MovCount'])
    if 'time' in source:
        columns_tokeep.extend(['CumTime', 'TimeSinceAcq', 'TimeSinceMov', 'CumFluo', 'CumCountFluo',
               'CumCine', 'CumCountCine', 'CumAcqTime', 'CumMovTime',
               'ShutterPositionXDcumsum', 'ShutterPositionXDcumcount',
               'ShutterPositionYDcumsum', 'ShutterPositionYDcumcount',
               'WedgeLeftDistanceDcumsum', 'WedgeLeftDistanceDcumcount',
               'WedgeLeftAngleDcumsum', 'WedgeLeftAngleDcumcount',
               'WedgeRightDistanceDcumsum', 'WedgeRightDistanceDcumcount',
               'WedgeRightAngleDcumsum', 'WedgeRightAngleDcumcount',
               'PositionCarmDcumsum', 'PositionCarmDcumcount',
               'PositionDetectorDcumsum', 'PositionDetectorDcumcount',
               'PositionPropellorDcumsum', 'PositionPropellorDcumcount',
               'FrontalBeamLongitudinalDcumsum', 'FrontalBeamLongitudinalDcumcount',
               'FrontalBeamTransversalDcumsum', 'FrontalBeamTransversalDcumcount',
               'FrontalRotateDetectorDcumsum', 'FrontalRotateDetectorDcumcount',
               'FrontalSwingDcumsum', 'FrontalSwingDcumcount',
               'FrontalZrotationDcumsum', 'FrontalZrotationDcumcount',
               'TableHeightDcumsum', 'TableHeightDcumcount', 'TableLateralDcumsum',
               'TableLateralDcumcount', 'TableLongitudinalDcumsum', 'AcqFreq', 'MovFreq', 
               'Rolling2MeanObjects', 'Rolling5MeanObjects',
                'Rolling10MeanObjects', 'Rolling5MeanPatient',
                'Rolling5MeanCardiologist'])
    if 'video' in source:
        columns_tokeep.extend(['#ObjectsInFrame', 'CardiologistCount', 'Lab AssistantCount',
               'PatientCount', 'PeopleCount', 'IsMovement', 'IsAcquisition',
               'IsCardiologist', 'IsPatient', 'IsLabAssistant', 'cell_0_1', 'cell_0_2',
               'cell_0_3', 'cell_0_4', 'cell_1_0', 'cell_1_1', 'cell_1_2', 'cell_1_3',
               'cell_1_4', 'cell_2_1', 'cell_2_2', 'cell_2_3', 'cell_2_4', 'cell_3_1',
               'cell_3_2', 'cell_3_3', 'cell_3_4', 'cell_4_2', 'cell_4_3', 'cell_4_4'])
    if 'all' in source:
        columns_tokeep=loaded_data[0].columns

    columns=loaded_data[0].columns
    if columns_to_keep!=None:
        columns_drop_1=[column for column in columns 
                    if (column not in columns_tokeep and column not in columns_to_keep)]
    else: columns_drop_1=[column for column in columns 
                if (column not in columns_tokeep)]
    #if source=='all':
     #   columns_drop_1=[None]
    if columns_drop!=None:
        for k in range(len(columns_drop)):
            columns_drop_1.append(columns_drop[k])
        
    if columns_drop_1 != None:
        for j in range(len(loaded_data)):
            loaded_data[j].drop(columns=columns_drop_1,inplace=True)
    
    
   # Y_ini = np.array([loaded_data[j]["Phase_num"] for j in range(len(loaded_data))]).astype(int) 

       
    X = np.array(loaded_data)
    columns=loaded_data[0].columns

    
    
    #X = loaded_data['data']
    #Y_ini = loaded_data['target']

    count_14_per_row_insec = np.sum(Y_ini == 14, axis=1)
    count_14_per_row_insec = 7573-count_14_per_row_insec
    count_14_per_row = count_14_per_row_insec/60
    indices_less = [index for index, value in enumerate(
        count_14_per_row) if value < cutoff[0] or value > cutoff[1]]
    Y_ini = np.delete(Y_ini, indices_less, axis=0)
    X = np.delete(X, indices_less, axis=0)

    if granularity == 'max':
        pass
    elif granularity == 'time_binary_1':
        Y_new = []
        for i in range(len(Y_ini)):
            # count how many 'end' seconds are present
            summa = np.array(Y_ini[i, :] == 14).sum()
           # Y_new.append(7573-summa)

            if summa >= 4873:  # procedure took <= 45 minuti
                Y_new.append(0)
            if 4273 <= summa < 4873:  # procedure took 45-55 minuti
                Y_new.append(1)
            if summa < 4273:  # procedure took more than 55 minuti
                Y_new.append(2)
        Y_ini = np.array(Y_new)
    elif granularity == 'time_binary_2':
        Y_new = []
        for i in range(len(Y_ini)):
            # count how many 'end' seconds are present
            summa = np.array(Y_ini[i, :] == 14).sum()
           # Y_new.append(7573-summa)

            if summa >= 4873:  # procedure took <= 45 minuti
                Y_new.append(0)
            if 4273 <= summa < 4873:  # procedure took 45-55 minuti
                Y_new.append(1)
            if summa < 4273:  # procedure took more than 55 minuti
                Y_new.append(2)
        Y_ini = np.array(Y_new)
    elif granularity == 'time_binary':
        Y_new = []
        for i in range(len(Y_ini)):
            # count how many 'end' seconds are present
            summa = np.array(Y_ini[i, :] == 14).sum()
           # Y_new.append(7573-summa)
            if summa >= 5713: #procedure took <= 31 min
                Y_new.append(0)
            if 5413 <= summa < 5713:  # procedure took 31-36 minuti
                Y_new.append(1)
                
            if 5113 <= summa < 5413:  # procedure took 37-41 minuti
                Y_new.append(2)
                
            if  4813 <= summa < 5113:  # procedure took 42-46 minuti
                Y_new.append(3)    
                
            if  4513 <= summa < 4813:  # procedure took 47-51 minuti
                Y_new.append(4) 
                
            if  3913 <= summa < 4513:  # procedure took 52-61 minuti
                Y_new.append(5) 
                
            if summa < 3913:  # procedure took more than 61 minuti
                Y_new.append(6)
       
        Y_ini = np.array(Y_new)
    elif granularity == 'time_binary_regression':
        Y_new = []
        for i in range(len(Y_ini)):
            # count how many 'end' seconds are present
            summa = np.array(Y_ini[i, :] == 14).sum()
            Y_new.append(7573-summa)
        Y_ini = np.array(Y_new)
    elif granularity == 'min':
        Y_ini[(Y_ini == 1) | (Y_ini == 2)] = 0  # C\A and C\B becomes C (0)
        Y_ini[(Y_ini == 3) | (Y_ini == 4) | (Y_ini == 5) | (
            Y_ini == 6)] = 1  # D,E,F,Fa become M_first (1)
        Y_ini[(Y_ini == 7) | (Y_ini == 8) | (Y_ini == 9) | (
            Y_ini == 10)] = 2  # G,H,Ha,I become M_second (2)
        Y_ini[(Y_ini == 11) | (Y_ini == 12) | (
            Y_ini == 13)] = 3  # L\J,L\K,L become L (3)
        Y_ini[(Y_ini == 14)] = 4  # End stays End (4)
    if granularity != 'time_binary_regression':
        if verbose:
            for i in range(len(np.unique(Y_ini))):
                print("Class {}: {} elements".format(np.unique(Y_ini)[i],Y_ini[Y_ini==np.unique(Y_ini)[i]].shape))

        NUM_CLASSES = len(np.unique(Y_ini))
        if NUM_CLASSES == 15:
            classes_final = np.array(
                ['C', 'C\A', 'C\B', 'D', 'E', 'F', 'Fa', 'G', 'Ha', 'H', 'I', 'L', 'L\J', 'L\K', 'End'])
            classes = ['C', 'C\A', 'C\B', 'D', 'E', 'F', 'Fa',
                       'G', 'Ha', 'H', 'I', 'L', 'L\J', 'L\K', 'End']
            phase_counts = pd.DataFrame(
                Y_ini[Y_ini != 14]).value_counts(normalize=True)
            phase_counts_withend = pd.DataFrame(
                Y_ini.reshape(-1)).value_counts(normalize=True, sort=False)
        elif NUM_CLASSES == 5:
            classes_final = np.array(['C', 'M_first', 'M_second', 'L', 'End'])
            classes = ['C', 'M_first', 'M_second', 'L', 'End']
            phase_counts = pd.DataFrame(
                Y_ini[Y_ini != 4]).value_counts(normalize=True)
            phase_counts_withend = pd.DataFrame(
                Y_ini.reshape(-1)).value_counts(normalize=True, sort=False)
        elif NUM_CLASSES == 3:
            classes_final = np.array(
                ['On time', '<=10 min delay', '>10 min delay'])
            classes = ['On time', '<=10 min delay', '>10 min delay']
            phase_counts = pd.DataFrame(
                Y_ini.reshape(-1)).value_counts(normalize=True, sort=False)
            phase_counts_withend = phase_counts
        elif NUM_CLASSES == 7:
            classes_final = np.array(
                ['<31','31-36', '37-41', '42-46','47-51','52-61','>=61'])
            classes = ['<31','31-36', '37-41', '42-46','47-51','52-61','>=61']
            phase_counts = pd.DataFrame(
                Y_ini.reshape(-1)).value_counts(normalize=True, sort=False)
            phase_counts_withend = phase_counts
        
        class_to_num_mapping = {class_label: i for i,
                                class_label in enumerate(classes_final)}
        X_new=np.zeros((X.shape[0],X.shape[1],X.shape[2]))
        for i in range(X.shape[0]):
            X_current=X[i,:count_14_per_row_insec[i],:]
            reshaped_array = X_current.reshape(-1, X.shape[-1])

            # Apply scaling
            scaler = MinMaxScaler()
            scaled_array = scaler.fit_transform(reshaped_array)

            # Reshape the scaled array back to its original shape
            X_new[i,:count_14_per_row_insec[i],:] = scaled_array
        # Assuming the last dimension is to be preserved
        reshaped_array = X.reshape(-1, X.shape[-1])

        # Apply scaling
        scaler = MinMaxScaler()
        scaled_array = scaler.fit_transform(reshaped_array)

        # Reshape the scaled array back to its original shape
        X = scaled_array.reshape(X.shape)
        X=X_new
        X=X[:,:cutoff[1]*60,:]
        X_list = []
        if granularity == 'max' or granularity == 'min':
            Y = np.zeros([Y_ini.shape[0], Y_ini.shape[1], NUM_CLASSES])
            for i in range(len(X)):
                Y[i] = to_categorical(Y_ini[i, :], num_classes=NUM_CLASSES)
            Y=Y[:,:cutoff[1]*60,:]

        elif granularity == 'time_binary':
            for j in range(5, cutoff[1], 5):
                X_list.append(X[:, :j*60, :])
            Y = np.zeros([Y_ini.shape[0], NUM_CLASSES])
            for i in range(len(X)):
                Y[i] = to_categorical(Y_ini[i], num_classes=NUM_CLASSES)
        
    elif granularity == 'time_binary_regression':
        # Assuming the last dimension is to be preserved
        reshaped_array = X.reshape(-1, X.shape[-1])

        # Apply scaling
        scaler = MinMaxScaler()
        scaled_array = scaler.fit_transform(reshaped_array)

        # Reshape the scaled array back to its original shape
        X = scaled_array.reshape(X.shape)

     #   X_list2 = []
      #  for i in range(X.shape[0]):
       #     X_list2.append(X[i, :count_14_per_row_insec[i], :])
        # X_list2_arr=np.array(X_list2)
        X_new=np.zeros((X.shape[0],X.shape[1],X.shape[2]))
        for i in range(X.shape[0]):
            X_current=X[i,:count_14_per_row_insec[i],:]
            reshaped_array = X_current.reshape(-1, X.shape[-1])

            # Apply scaling
            scaler = MinMaxScaler()
            scaled_array = scaler.fit_transform(reshaped_array)

            # Reshape the scaled array back to its original shape
            X_new[i,:count_14_per_row_insec[i],:] = scaled_array
        X=X_new
        scaler_Y = MinMaxScaler()
        scaled_array_Y = scaler_Y.fit_transform(Y_ini.reshape(-1, 1))
       # scaled_array_Y=Y_ini
        # Reshape the scaled array back to its original shape
        Y = scaled_array_Y
       # Y=Y_ini
        X_list = []
        # X_list_out=[]
        for j in range(5, cutoff[1], 5):
            # X_toappend=X[:]
            X_list.append(X[:, :j*60, :].copy())
            # X_list2_big=[]

           # for h in range(len(X_list2)):
           #     X_list2_big.append(X_list2[h][:j*60,:])
            # X_list_out.append(X_list2_big)
        phase_counts = pd.DataFrame(
            Y_ini.reshape(-1)).value_counts(normalize=True, sort=False)
        phase_counts_withend = phase_counts
       # X_list=X_list_out
    # ==========================================================================================================================
    # ==========================================================================================================================
    #X_list=[X_list[-2],X_list[-1]]
    # Model's creation
    # Create callbacks
    if granularity=='time_binary':
        for j in range(X_phase_only.shape[0]):
            h=0
           # for h in range(X_phase_only.shape[1]):
            while X_phase_only[j,h,1]!=0:
                X_phase_only[j,h,1]=h+1
                h+=1
    if reduce_dim:
        X=X[:,0::reduce_dim,:]
        X_phase_only=X_phase_only[:,0::reduce_dim,:]
        Y=Y[:,0::reduce_dim,:]
        #Y=Y[:,0::reduce_dim,:]
    #X_list=[X_list[-1]]
    # Custom loss function
    inverseN = 1 / len(phase_counts_withend)
    if granularity == 'min':
        weights = np.array([5, 3.5, 2.5, 4.9, 0.01])
        weights = np.array([inverseN/phase for phase in phase_counts_withend])

    elif granularity == 'max':
        weights = np.array([inverseN/phase for phase in phase_counts_withend])
    elif granularity == 'time_binary':
        weights = np.array([inverseN/phase for phase in phase_counts_withend])
       # weights=np.array([1.6,2,25,3,25,1.63,30,8.4,12.26,1.32,4.79,7.06,2.51,13.61,0.10])
    if granularity=='time_binary_regression':
        return X,Y,X_list,weights,classes,classes_final,class_to_num_mapping,phase_counts,phase_counts_withend,scaler_Y,columns
    else:  return X,Y,X_phase_only,X_list,weights,classes,classes_final,class_to_num_mapping,phase_counts,phase_counts_withend,columns

#X,Y,X_list,weights,classes,classes_final,class_to_num_mapping,phase_counts,phase_counts_withend,columns=prepare_data(pc='PC_mine',granularity='max',cutoff=[0,130],verbose=0,columns_to_drop=None,columns_to_keep='cell_1_0',source='all')
