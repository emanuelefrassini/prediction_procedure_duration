import os
import datetime
def folder_creation(pc,granularity):
    if pc=='PC_mine':
        base_folder = r'C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Codes\MTSC\runs\\'
    elif pc=='PC_tudelft' or pc=='PC_rdg':
        base_folder = r'C:\Users\efrassini\OneDrive - Delft University of Technology\Work\Codes\MTSC\runs\\'

    # Get the current date and time
    current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create the folder name with the current date and time
    folder_name = f'runs_{current_datetime}_{granularity}'

    # Path for the new folder
    folder_path_inforun = os.path.join(base_folder, folder_name)

    # Info run save path
  #  os.makedirs(folder_path_inforun, exist_ok=True)


    # Model save path
    folder_path_save_model = os.path.join(folder_path_inforun, 'model')
   # os.makedirs(folder_path_save_model, exist_ok=True)

    # Training plots save path
    folder_path_training = os.path.join(folder_path_inforun, 'train')
   # os.makedirs(folder_path_training, exist_ok=True)

    # Test plots save path
    folder_path_test = os.path.join(folder_path_inforun, 'test')
   # os.makedirs(folder_path_test, exist_ok=True)

    # Define the log directory for TensorBoard
    folder_path_tensorboard = os.path.join(folder_path_inforun, 'log_tensorboard')
   # os.makedirs(folder_path_tensorboard, exist_ok=True)
    
    return folder_path_inforun,folder_path_save_model,folder_path_training,folder_path_test,folder_path_tensorboard