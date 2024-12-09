
import numpy as np
from sklearn.metrics import confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
#import seaborn as sn
from utils.model_creation import model_creation
from utils.data_preparation import prepare_data
import tensorflow as tf

def fit_model(X,Y,X_list,pc,granularity,method,error_margin,UNITS,DROPOUT_RATE,LEARNING_RATE,EPOCHS,BATCH_SIZE,loss,weights,classes,folder_path_inforun,
              folder_path_save_model,folder_path_training,folder_path_test,folder_path_tensorboard,NUM_FOLDS=3,cutoff=[0,130],majority_vote=0,columns_to_drop=None,columns_to_keep=None,SOURCE='all',MAXLENPADDED=[7573],REDUCE_DIM=None,num_transf_blocks=8,head_size=4,ffd=4,dropout_enc=0.25):
    if granularity=='time_binary_regression':
        X,Y,X_list,weights,classes,classes_final,class_to_num_mapping,phase_counts,phase_counts_withend,scaler_Y,columns=prepare_data(pc,granularity,cutoff=[0,130],verbose=0,columns_to_drop=columns_to_drop,columns_to_keep=columns_to_keep,source=SOURCE,reduce_dim=REDUCE_DIM)
    else:  
        X,Y,X_phase_only,X_list,weights,classes,classes_final,class_to_num_mapping,phase_counts,phase_counts_withend,columns=prepare_data(pc,granularity,cutoff,verbose=0,columns_to_drop=columns_to_drop,columns_to_keep=columns_to_keep,source=SOURCE,reduce_dim=REDUCE_DIM)
    if granularity=='min':
        NUM_CLASSES=5
    elif granularity=='max':
        NUM_CLASSES=15
    elif granularity=='time_binary':
        NUM_CLASSES=7
        Y4=np.zeros((X.shape[0],X.shape[1],NUM_CLASSES))
        for h in range(Y4.shape[1]):
            Y4[:,h,:]=Y
        Y=Y4
        print(Y.shape)
        #if REDUCE_DIM:
         #   Y=Y[:,0::REDUCE_DIM,:]
          #  print(Y.shape)
    


    if granularity == 'min' or granularity == 'max':
        start=time.time()
        # Experiment set up
        kfold = KFold(n_splits=NUM_FOLDS, shuffle=True)
        loss_train_per_fold = []
        accuracy_train_per_fold = []

        cm_for_std = np.zeros(
            (len(error_margin), NUM_FOLDS, NUM_CLASSES, NUM_CLASSES))

        loss_test_per_fold = []
        accuracy_test_per_fold = []

        time_fold = []

        fold = 0
        cm_list = []
        accuracy_per_class_list = []
        std_per_class = []
        cm = np.zeros([NUM_CLASSES, NUM_CLASSES])
        for k in range(len(error_margin)):
            cm_list.append(cm)
        for train, test in kfold.split(X):
            start_fold = time.time()

            fold += 1
            print("\n Start training on fold number {} out of {}".format(
                fold, NUM_FOLDS))
            X_train = X[train, :, :]
            y_train = Y[train, :]
            X_test = X[test, :, :]
            y_test = Y[test, :]

           # model = create_model()
            model,model_callbacks=model_creation(X,Y,granularity,method,UNITS,DROPOUT_RATE,LEARNING_RATE,EPOCHS,loss,weights,folder_path_tensorboard,num_transf_blocks,head_size,ffd,dropout_enc)

            history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2, shuffle=False,
                                callbacks=model_callbacks, validation_data=(X_test, y_test))
            print("Fold {} took {} seconds".format(
                fold, time.time()-start_fold))

            loss_train_per_fold.append(np.mean(history.history['loss']))
            accuracy_train_per_fold.append(
                np.mean(history.history['categorical_accuracy']))
            print("\n---------------------------------------------------------")

            y_pred = model.predict(X_test)

            if granularity == 'max' or granularity == 'min':
                Y_test = np.argmax(y_test, axis=2)  # Convert one-hot to index
                y_pred = np.argmax(y_pred, axis=2)
                if len(error_margin) > 0:
                    y_pred_dic = {}

                    for k in range(len(error_margin)):
                        y_temp = y_pred.copy()
                       # print("\n y_temp before is {}".format(y_temp))
                        iscorrect = 0
                        for row in range(y_pred.shape[0]):
                            for i in range(y_pred.shape[1]-error_margin[k]-1):
                                iscorrect = 0
                                if Y_test[row, i] != Y_test[row, i+1]:

                                    for j in range(i, i+error_margin[k]):

                                        if y_temp[row, j] == Y_test[row, i]:
                                            iscorrect = 1
                                            break
                                    if iscorrect:
                                        y_temp[row, i:i+j] = Y_test[row, i+1]
                                       # y_temp[row, i] = Y_test[row, i+1]

                                     #   print("\n sono dentro")
                        # print("\n y_temp after is {}".format(y_temp))
                        # y_temp=[]
                        # a.append(y_temp)
                        # print("\n a is {}".format(a))

                        y_pred_dic["y_pred_{}".format(k)] = y_temp.copy()
                        # y_pred_dic={}
                        # for k in range(len(error_margin)-1):
                        #     y_pred_dic["y_pred_{}".format(k)]=y_temp*k
                        #     print((y_pred_dic["y_pred_{}".format(k)]-y_pred_dic["y_pred_{}".format(k+1)]).sum())

                if majority_vote > 0:
                    # Majority vote
                    for i in range(0, y_pred.shape[1]-73, majority_vote):
                        start_index = i
                        end_index = i+majority_vote

                        predictions = y_pred[:, start_index:end_index]
                        votes_array = np.zeros(
                            [predictions.shape[0], majority_vote])

                        for j in range(predictions.shape[0]):

                            unique_elements, counts = np.unique(
                                predictions[j, :], return_counts=True)

                        # Find the most common element and its count
                            most_common_index = np.argmax(counts)
                            votes = unique_elements[most_common_index]
                       # votes = np.max(predictions,axis=1)
                            votes_array[j, :] = np.full(majority_vote, votes)
                        y_pred[:, start_index:end_index] = votes_array
            
            elif granularity == 'time_binary':
                Y_test = np.argmax(y_test, axis=1)  # Convert one-hot to index
                y_pred = np.argmax(y_pred, axis=1)

            print(classification_report(Y_test.flatten(),
                  y_pred.flatten(), target_names=classes))
            # cm+=confusion_matrix(Y_test.flatten(), y_pred.flatten())
            for k in range(len(error_margin)):
                #  print((y_pred_dic["y_pred_{}".format(k)]-y_pred_dic["y_pred_{}".format(k+1)]).sum())
                cm_temp = confusion_matrix(
                    Y_test.flatten(), y_pred_dic["y_pred_{}".format(k)].flatten()).copy()
                cm_for_std[k, fold-1, :, :] = cm_temp.copy()
                cm_list[k] = cm_list[k]+cm_temp
               # cm_list[k]+=confusion_matrix(Y_test.flatten(), y_pred_dic["y_pred_{}".format(k)].flatten())

            evaluation = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)

            print("\n-----------------------------------------------------")
            print(
                "\nTest Set Evaluation - Loss: {:.4f}, Accuracy: {:.4f}".format(evaluation[0], evaluation[1]))
            loss_test_per_fold.append(evaluation[0])
            accuracy_test_per_fold.append(evaluation[1])

           # keras.backend.clear_session()
            time_fold.append(time.time()-start_fold)

        for k in range(len(error_margin)):
            cm = cm_list[k]
            accuracy_per_class = cm.diagonal()/cm.sum(axis=1)
            accuracy_per_class_list.append(accuracy_per_class)
            diag = np.zeros(shape=(NUM_FOLDS, NUM_CLASSES))
            for indice in range(cm_for_std.shape[1]):
                diag[indice, :] = cm_for_std[k, indice, :, :].diagonal(
                )/cm_for_std[k, indice, :, :].sum(axis=1)
            std_matrices = np.std(diag, axis=0)
            std_per_class.append(std_matrices)
            
        accuracy_per_class_0 = accuracy_per_class_list[0]
        accuracy_per_class_5 = accuracy_per_class_list[1]
        if granularity == 'max':
            result_array_0 = phase_counts * np.delete(accuracy_per_class_0, 14)
            result_array_5 = phase_counts * np.delete(accuracy_per_class_5, 14)

        elif granularity == 'min':
            result_array_0 = phase_counts * np.delete(accuracy_per_class_0, 4)
            result_array_5 = phase_counts * np.delete(accuracy_per_class_5, 4)


        # Sum the resulting array
        total_sum_0 = np.sum(result_array_0)
        total_sum_5 = np.sum(result_array_5)

        
        return total_sum_0,total_sum_5
r'''
        # ==========================================================================================================================
        # ==========================================================================================================================
        # Files saving
        model.save(os.path.join(folder_path_save_model, 'model.keras'))

        # Save the plots
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.plot(loss_train_per_fold, label='Loss', color='red', marker='o')
        ax1.set_title('Loss vs. Folds')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax2.plot(accuracy_train_per_fold, label='Accuracy', marker='o')
        ax2.set_title('Accuracy vs. Folds')
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        # Save the figure
        plot_file_path = os.path.join(
            folder_path_training, 'loss_accuracy_train_kfold.png')
        plt.savefig(plot_file_path)
        plt.close()

        file_path = os.path.join(folder_path_training,
                                 'accuracy_train_kfold.txt')
        with open(file_path, 'w') as file:
            for item in accuracy_train_per_fold:
                file.write("%s\n" % item)
            file.write(
                f"Average accuracy on train set is: {np.mean(accuracy_train_per_fold)} +- {np.std(accuracy_train_per_fold)}")

        file_path = os.path.join(folder_path_training, 'loss_train_kfold.txt')
        with open(file_path, 'w') as file:
            for item in loss_train_per_fold:
                file.write("%s\n" % item)
            file.write(
                f"Average loss on train set is: {np.mean(loss_train_per_fold)}  +- {np.std(loss_train_per_fold)}")

        print(f'\nTraining plots and files saved to {plot_file_path}')

        # Save the plots
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.plot(loss_test_per_fold, label='Loss', color='red', marker='o')
        ax1.set_title('Loss per fold in test set')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax2.plot(accuracy_test_per_fold, label='Accuracy', marker='o')
        ax2.set_title('Accuracy per fold in test set')
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        plt.tight_layout()  # Adjusts subplot parameters to give specified padding
        # plt.show()

        # Save the figure
        plot_file_path = os.path.join(
            folder_path_test, 'loss_accuracy_test_kfold.png')
        plt.savefig(plot_file_path)
        plt.close()
        # plt.show()

        # fig = plt.figure()

        for k in range(len(error_margin)):
            cm = cm_list[k]
            cm_normalized = cm / cm.sum(axis=1, keepdims=True)

            df_cm = pd.DataFrame(
                cm_normalized[:-1, :-1], index=classes[:-1], columns=classes[:-1])

            plt.figure(figsize=(20, 12), dpi=100)

            # Create the heatmap with custom parameters
            ax = sn.heatmap(
                df_cm, annot=cm[:-1, :-1], cmap='Oranges',  annot_kws={"size": 10})

            # Save the figure to a PNG file
            plot_file_path = os.path.join(
                folder_path_inforun, 'confusion_matrix_error_{}.png'.format(error_margin[k]))

            # Save the heatmap to the specified file
            plt.savefig(plot_file_path)
            plt.close()
        # Show the plot (optional)
       # plt.show()

        file_path = os.path.join(folder_path_test, 'accuracy_test.txt')
        with open(file_path, 'w') as file:
            for item in accuracy_test_per_fold:
                file.write("%s\n" % item)
            file.write(
                f"Average accuracy on test set is: {np.mean(accuracy_test_per_fold)}  +- {np.std(accuracy_test_per_fold)}")

        file_path = os.path.join(folder_path_test, 'loss_test.txt')
        with open(file_path, 'w') as file:
            for item in loss_test_per_fold:
                file.write("%s\n" % item)
            file.write(
                f"Average loss on test set is: {np.mean(loss_test_per_fold)}  +- {np.std(loss_test_per_fold)}")

        file_path = os.path.join(folder_path_test, 'confusion_matrix.txt')
        with open(file_path, 'w') as f:
            for line in cm:
                np.savetxt(f, line, fmt='%.2f')

        file_path = os.path.join(folder_path_inforun, 'info_run.txt')
        with open(file_path, 'w') as file:
            file.write("Weights: \n")
            for classes_name, item in zip(classes, weights):
                file.write("Class {}: {}  -   ".format(classes_name, item))
            file.write("\n\nEpochs: {} \n".format(EPOCHS))
            file.write("Dropout rate: {} \n".format(DROPOUT_RATE))
            file.write("Batch size: {} \n".format(BATCH_SIZE))
            file.write("Learning rate: {} \n".format(LEARNING_RATE))
            file.write("Units: {} \n".format(UNITS))
            file.write("Number of folds: {} \n".format(NUM_FOLDS))
            file.write("Cutoff: {}-{} \n".format(cutoff[0], cutoff[1]))
            file.write("Majority vote: {} \n\n".format(majority_vote))
            file.write("Total time: {} s \n".format(str(time.time() - start)))
            file.write("Time per fold:\n")
            for item in time_fold:
                file.write("%s s - " % item)
       # file_path = os.path.join(folder_path_inforun, 'error_margin.txt')
       # with open(file_path, 'w') as file:
       #     file.write("%s\n" % error_margin)

        file_path = os.path.join(folder_path_inforun, 'model_summary.txt')

        def myprint(s):
            with open(file_path, 'a') as f:
                print(s, file=f)
        model.summary(print_fn=myprint)

        for k in range(len(error_margin)):
            accuracy_per_class = accuracy_per_class_list[k]
            std_per_class_single = std_per_class[k]
            if granularity == 'max':
                result_array = phase_counts * np.delete(accuracy_per_class, 14)
            elif granularity == 'min':
                result_array = phase_counts * np.delete(accuracy_per_class, 4)

            # Sum the resulting array
            total_sum = np.sum(result_array)

            file_path = os.path.join(
                folder_path_inforun, 'accuracy_per_class_error_{}.txt'.format(error_margin[k]))
            # Open the file in write mode
            with open(file_path, 'w') as file:
                # Iterate through classes and accuracy_per_class simultaneously
                for class_name, accuracy, stdev in zip(classes, accuracy_per_class, std_per_class_single):

                    # Write the data in the specified format to the file
                    file.write(
                        f"Phase {class_name} - accuracy: {accuracy} +- {stdev}\n")
                file.write(
                    f"Average accuracy among all the classes: {np.mean(accuracy_per_class)}")
                file.write(
                    f"\nWeighted average accuracy among all the classes: {total_sum}")
                file.write(f"\nError margin: {error_margin[k]}")

        print(f'\nTest plots and files saved to {plot_file_path}')

        plt.figure(figsize=(10, 6))
        bars = plt.bar(classes[:-1], phase_counts, width=0.4)

        plt.xlabel('Class')
        plt.ylabel('Abundance')
        plt.title('Abundance Distribution of {} Classes'.format(NUM_CLASSES))

        # Adding annotations on top of the bars
        for bar, count in zip(bars, phase_counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), np.round(count, 3),
                     ha='center', va='bottom', fontsize=8)

        plot_file_path = os.path.join(
            folder_path_inforun, 'abbundance_per_class.png')

        plt.savefig(plot_file_path)  # Save the heatmap to the specified file
        plt.close()
        

    elif granularity == 'time_binary':
        
        accuracy_per_slice = []
        std_accuracy_per_slice = []
        time_per_slice = []
        fold=0
    # Experiment set up
        kfold = KFold(n_splits=NUM_FOLDS, shuffle=True)
        loss_train_per_fold = []
        accuracy_train_per_fold = []

        loss_test_per_fold = np.zeros((len(MAXLENPADDED),NUM_FOLDS))
        accuracy_test_per_fold = np.zeros((len(MAXLENPADDED),NUM_FOLDS))
        accuracy_test_per_fold=[]
        time_fold = []
       
        cm_list = []
        accuracy_per_class_list = []
        start_fold = time.time()
        #Xcopy=X
        Xcopy=X_phase_only
       
        num=0
        print(Y.shape)
        for ind in range(len(MAXLENPADDED)):
            if max(MAXLENPADDED)>X.shape[1]:
                MAXLENPADDED=[X.shape[1]]
            X=tf.keras.utils.pad_sequences(Xcopy,padding='post',maxlen=MAXLENPADDED[ind],truncating='post',dtype='float32')
            num+=1
            print(f"\nStart training on slice number {num} out of {len(MAXLENPADDED)}")
            fold=0
            for train, test in kfold.split(X):
                fold += 1
                print(f"\nStart training on fold number {fold} out of {NUM_FOLDS}")
                loss_test_per_slice=[]
                accuracy_test_per_slice=[]
                loss_train_per_slice=[]
                accuracy_train_per_slice=[]
                time_slice=[]
                X_list=X #new
        
                start_slice = time.time()
                print("\n=============================================================")
                print("\n=============================================================")

               # print("\nStart training on slice number {} out of {}".format(
                #    ind, len(X_list))) #new

               # X = X_list[ind] #new
            

                #  print("\n Start training on fold number {} out of {}".format(fold,NUM_FOLDS))

                X_train = X[train, :, :]
                y_train = Y[train, :]
                X_test = X[test, :, :]
                y_test = Y[test, :]
                
                model,model_callbacks=model_creation(X,Y,granularity,method,UNITS,DROPOUT_RATE,LEARNING_RATE,EPOCHS,loss,weights,folder_path_tensorboard)
                print("X_train {} , y_train {})".format(X_train.shape,y_train.shape))
                print("X_test {} , y_test {})".format(X_test.shape,y_test.shape))

                history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, shuffle=False,
                                    callbacks=model_callbacks, validation_data=(X_test, y_test))
                print("\nFold {} took {} seconds".format(
                    fold, time.time()-start_fold))

               # loss_train_per_slice.append(np.mean(history.history['loss']))
               # accuracy_train_per_slice.append(
               #    np.mean(history.history['categorical_accuracy']))
                print("\n---------------------------------------------------------")

                evaluation = model.evaluate(
                    X_test, y_test, batch_size=BATCH_SIZE)
                evaluation_train=model.evaluate(X_train, y_train, batch_size=BATCH_SIZE)
                print("\nSlice number {} Evaluation - Train: Loss: {:.4f}, Accuracy: {:.4f} \n Test: Loss: {:.4f}, Accuracy: {:.4f}".format(
                    ind, evaluation_train[0],evaluation_train[1], evaluation[0],evaluation[1]))
                
                
                loss_test_per_slice.append(evaluation[0])
                accuracy_test_per_slice.append(evaluation[1])
                
                loss_train_per_slice.append(evaluation_train[0])
                accuracy_train_per_slice.append(evaluation_train[1])
                

                time_slice.append(time.time()-start_fold)
                y_pred=model.predict(X_test,batch_size=BATCH_SIZE)
                y_pred=np.argmax(y_pred,axis=2)
                y_test=np.argmax(y_test,axis=2)
   
                file_path2 = os.path.join(folder_path_inforun, "test_predictions")
                os.makedirs(file_path2, exist_ok=True)
                file_path2 = os.path.join(file_path2, f"test_slice_{ind}.txt")
    
                with open(file_path2, 'w') as file:
                    for truth, prediction in zip(y_test, y_pred):
                        file.write("Truth: {}, prediction: {}  \n".format(truth,prediction))
                
           # accuracy_per_fold.append(np.mean(accuracy_test_per_fold))
            
           # accuracy_test_per_fold[:,fold-1]=np.array(accuracy_test_per_slice)
           # loss_test_per_fold[:,fold-1]=np.array(loss_test_per_slice)
            accuracy_test_per_fold.append(np.mean(accuracy_test_per_slice))
            accuracy_train_per_fold.append(np.mean(accuracy_train_per_slice))

           # std_accuracy_per_slice.append(np.std(accuracy_test_per_fold,axis=0))
            time_per_slice.append(time.time()-start_slice)
               # print("\n-----------------------------------------------------")
               # print("\nTest Set Evaluation - Loss: {:.4f}, Accuracy: {:.4f}".format(evaluation[0], evaluation[1]))
             
            
        # keras.backend.clear_session()
        # Create x-axis values (assuming indices as x-axis)
        accuracy_per_slice=np.mean(accuracy_test_per_fold,axis=0)   
        # Generate ticks: 5, 10, 15, 20, ...
        #custom_ticks = [5 * (i+1) for i in range(accuracy_test_per_fold.shape[0])]
        custom_ticks=MAXLENPADDED
        x_values = custom_ticks
        # x_values = range(len(accuracy_per_slice))
    
        # Plotting the list
        plt.plot(x_values, accuracy_test_per_fold, marker='o', linestyle='-')
        plt.xlabel('Elapsed time [s]')
        plt.ylabel('Accuracy')
        plt.title('Accuracy in the prediction of the duration of the procedure')
    
        plot_file_path = os.path.join(
            folder_path_inforun, 'accuracy_per_slice.png')
        
        plt.plot(x_values, accuracy_train_per_fold, marker='o', linestyle='-')
        plt.xlabel('Elapsed time [s]')
        plt.ylabel('Accuracy')
        plt.title('Accuracy training in the prediction of the duration of the procedure')
    
        plot_file_path = os.path.join(
            folder_path_inforun, 'accuracy_per_slice_training.png')
    
    
        plt.savefig(plot_file_path)  # Save the heatmap to the specified file
        plt.close()
    
        plt.figure(figsize=(10, 6))
        bars = plt.bar(classes, phase_counts, width=0.4)
    
        plt.xlabel('Class')
        plt.ylabel('Abundance')
        plt.title('Abundance Distribution of {} Classes'.format(NUM_CLASSES))
    
        # Adding annotations on top of the bars
        for bar, count in zip(bars, phase_counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), np.round(count, 3),
                     ha='center', va='bottom', fontsize=8)
    
        plot_file_path = os.path.join(
            folder_path_inforun, 'abbundance_per_class.png')
    
        plt.savefig(plot_file_path)  # Save the heatmap to the specified file
        plt.close()
    
        file_path = os.path.join(folder_path_inforun, 'time.txt')
        with open(file_path, 'w') as file:
            for item in time_per_slice:
                file.write("%s\n" % item)
        
        file_path = os.path.join(folder_path_inforun, 'info_run.txt')
        with open(file_path, 'w') as file:
            file.write("Weights: \n")
            for classes_name, item in zip(classes, weights):
                file.write("Class {}: {}  -   ".format(classes_name, item))
            file.write("\n\nEpochs: {} \n".format(EPOCHS))
            file.write("Dropout rate: {} \n".format(DROPOUT_RATE))
            file.write("Batch size: {} \n".format(BATCH_SIZE))
            file.write("Learning rate: {} \n".format(LEARNING_RATE))
            file.write("Units: {} \n".format(UNITS))
            file.write("Number of folds: {} \n".format(NUM_FOLDS))
            file.write("Cutoff: {}-{} \n".format(cutoff[0], cutoff[1]))
            file.write("Majority vote: {} \n\n".format(majority_vote))
            #file.write("Total time: {} s \n".format(str(time.time() - start)))
            file.write("Time per fold:\n")
            for item in time_fold:
                file.write("%s s - " % item)
    
        #file_path = os.path.join(folder_path_inforun, 'accuracy_per_slice.txt')
        #with open(file_path, 'w') as file:
         #   for index, item in enumerate(accuracy_per_slice):
          #      file.write("Slice number {} - accuracy: {} +- {}\n".format(index,
           #                item, std_accuracy_per_slice[index]))

    elif granularity == 'time_binary_regression':
        accuracy_per_slice = []
        std_accuracy_per_slice = []
        time_per_slice = []
        for ind in range(len(X_list)):
            start_slice = time.time()
            print("\n=============================================================")
            print("\n=============================================================")

            print("\nStart training on slice number {} out of {}".format(
                ind, len(X_list)))

            X = X_list[ind]
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

            for train, test in kfold.split(X):
                start_fold = time.time()

                fold += 1
                #  print("\n Start training on fold number {} out of {}".format(fold,NUM_FOLDS))

                # X_train=[X[i] for i in (train)]
                # X_test=[X[i] for i in (test)]
                X_train = X[train, :, :]
                y_train = Y[train]
                X_test = X[test, :, :]
                y_test = Y[test]

                model,model_callbacks=model_creation(X,Y,granularity,method,UNITS,DROPOUT_RATE,LEARNING_RATE,EPOCHS,loss,weights,folder_path_tensorboard)

                history = model.fit(X_train, y_train, batch_size=1, epochs=EPOCHS, verbose=1, shuffle=False,
                                    callbacks=model_callbacks, validation_data=(X_test, y_test))

                print("\nFold {} took {} seconds".format(
                    fold, time.time()-start_fold))

                print("\n---------------------------------------------------------")

                y_train_predict = model.predict(X_train)
                y_test_predict = model.predict(X_test)

                y_train_predict_rescaled = scaler_Y.inverse_transform(
                    y_train_predict)
                y_train_rescaled = scaler_Y.inverse_transform(y_train)

                y_test_predict_rescaled = scaler_Y.inverse_transform(
                    y_test_predict)
                y_test_rescaled = scaler_Y.inverse_transform(y_test)
                # calculate root mean squared error
                trainScore = np.sqrt(
                    (mean_squared_error(y_train_rescaled, y_train_predict_rescaled)))
                # trainScore =np.sqrt((mean_squared_error(y_train, y_train_predict)))

               # print('Train Score: %.2f RMSE' % (trainScore))
                testScore = np.sqrt(mean_squared_error(
                    y_test_rescaled, y_test_predict_rescaled))
                # testScore = np.sqrt(mean_squared_error(y_test,y_test_predict))

                accuracy_train_per_fold.append(np.mean(trainScore))
                std_train_per_fold.append(np.std(trainScore))

               # accuracy_train_per_fold.append(np.mean(history.history['categorical_accuracy']))
               # print("\n-----------------------------------------------------")
               # print("\nTest Set Evaluation - Loss: {:.4f}, Accuracy: {:.4f}".format(evaluation[0], evaluation[1]))
               # loss_test_per_fold.append(evaluation[0])
                accuracy_test_per_fold.append(np.mean(testScore))
                std_test_per_fold.append(np.std(testScore))

                time_fold.append(time.time()-start_fold)
            print("\nSlice number {} Evaluation -  Accuracy: {:.4f}".format(ind,
                  np.mean(accuracy_test_per_fold),))
            accuracy_per_slice.append(np.mean(accuracy_test_per_fold))
            std_accuracy_per_slice.append(np.std(accuracy_test_per_fold))
            time_per_slice.append(time.time()-start_slice)
            # keras.backend.clear_session()
            # Create x-axis values (assuming indices as x-axis)
        # Generate ticks: 5, 10, 15, 20, ...
        custom_ticks = [5 * (i+1) for i in range(len(accuracy_per_slice))]

        x_values = custom_ticks
        # x_values = range(len(accuracy_per_slice))

        # Plotting the list
        plt.plot(x_values, accuracy_per_slice, marker='o', linestyle='-')
        plt.xlabel('Elapsed time')
        plt.ylabel('mae [s]')
        plt.title('Error in the prediction of the duration of the procedure')

        plot_file_path = os.path.join(folder_path_inforun, 'mae_per_slice.png')

        plt.savefig(plot_file_path)  # Save the heatmap to the specified file
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

        file_path = os.path.join(folder_path_inforun, 'accuracy_per_slice.txt')
        with open(file_path, 'w') as file:
            for index, item in enumerate(accuracy_per_slice):
                file.write("Slice number {} - accuracy: {} +- {}\n".format(index,
                           item, std_accuracy_per_slice[index]))
    print("\nEverything has been saved succesfully!")
'''   