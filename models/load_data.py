import numpy as np

def load_data(file_path= r'C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Codes\MTSC\time_binary_regression\data\dataset.npz'):

    loaded_data = np.load(file_path)

    # Retrieve arrays X and Y from loaded data
    X = loaded_data['X']
    Y = loaded_data['Y']
    return X,Y
X,Y=load_data()

