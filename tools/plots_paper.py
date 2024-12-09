# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:38:50 2024

@author: emanuelefrassi
"""

import pandas as pd
import matplotlib.pyplot as plt



file_path=r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Codes\MTSC\runs\runs_PAPER_20240429-121956\summary.xlsx"
df = pd.read_excel(file_path, usecols="A, B, C,D,E,H,I,J,K,N,O,P,Q,T,U,V,W,Z,AA,AB,AC,AE,AF,AG,AH,AI,AJ,AL,AM,AN,AO,AP,AQ,AR,AS,AT,AU,AW,AX,AY,AZ,BA,BB,BC,BD,BE,BF,BH,BI,BJ,BK,BL")

def rename_columns(col):
    if col.endswith('.1'):
        return col.replace('.1', '_10')
    elif col.endswith('.2'):
        return col.replace('.2', '_30')
    elif col.endswith('.3'):
        return col.replace('.3', '_60')
    elif col.endswith('.4'):
        return col.replace('.4', '_120')
    return col
df.rename(columns=lambda col: rename_columns(col), inplace=True)
columns = df.columns
df = df.rename(columns={columns[1]: 'MAE_5', columns[2]: 'SMAPE_5',columns[3]:'Training time_5',columns[4]:'Testing time_5'})



# Select the columns to plot
columns_time = [col for col in df.columns if col.startswith('Training')]
columns_error=[col for col in df.columns if col.startswith('Error_training')]



plot_time = df[columns_time]/60
plot_error=df[columns_error]/60

# Plot the data with error bars
ax = plot_time.plot(
    kind='bar',
    figsize=(10, 6),
    yerr=plot_error.values.T,  # Transpose to match the shape
    capsize=5,                 # Add caps to the error bars
    error_kw={'elinewidth': 1, 'ecolor': 'black'}  # Error bars style
)

# Customize plot appearance
plt.title('1-fold training time for the models')
plt.xlabel('Model')
plt.ylabel('Time [min]')
plt.xticks(
    ticks=range(len(df.index)), 
    labels=df['Model'], 
    rotation=45
)
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
plt.grid(axis='y', linestyle='--', linewidth=0.7)

plt.show()





#columns_to_plot = ['Training time', 'Testing time', 'Training time_10', 'Testing time_10']

# Select the columns to plot
columns_time = [col for col in df.columns if col.startswith('Testing')]
columns_error=[col for col in df.columns if col.startswith('Error_testing')]

plot_time = df[columns_time]
plot_error=df[columns_error]



# Plot the data with error bars
ax = plot_time.plot(
    kind='bar',
    figsize=(10, 6),
    yerr=plot_error.values.T,  # Transpose to match the shape
    capsize=5,                 # Add caps to the error bars
    error_kw={'elinewidth': 1, 'ecolor': 'black'}  # Error bars style
)

# Customize plot appearance
plt.title('1-fold testing time for the models')
plt.xlabel('Model')
plt.ylabel('Time [sec]')
plt.xticks(
    ticks=range(len(df.index)), 
    labels=df['Model'], 
    rotation=45
)
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
plt.grid(axis='y', linestyle='--', linewidth=0.7)

plt.show()










# Plot the data
columns_mae = [col for col in df.columns if col.startswith('MAE')]
columns_error=[col for col in df.columns if col.startswith('Error_MAE')]

plot_mae = df[columns_mae]
plot_error = df[columns_error]

#plot_mae.plot(kind='bar', figsize=(10, 6))
ax = plot_mae.plot(
    kind='bar',
    figsize=(10, 6),
    yerr=plot_error.values.T,  # Transpose to match the shape
    capsize=5,                 # Add caps to the error bars
    error_kw={'elinewidth': 1, 'ecolor': 'black'}  # Error bars style
)
plt.title('Mean absolute error in the prediction of duration')
plt.xlabel('Model')
plt.ylabel('MAE [min]')
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
plt.legend(title='Metrics')
plt.grid(axis='y', linestyle='--', linewidth=0.7)

plt.show()




columns_smape = [col for col in df.columns if col.startswith('SMAPE')]
columns_error=[col for col in df.columns if col.startswith('Error_SMAPE')]

plot_mae = df[columns_smape]
plot_error = df[columns_error]

# Plot the data
#plot_mae.plot(kind='bar', figsize=(10, 6))
ax = plot_mae.plot(
    kind='bar',
    figsize=(10, 6),
    yerr=plot_error.values.T,  # Transpose to match the shape
    capsize=5,                 # Add caps to the error bars
    error_kw={'elinewidth': 1, 'ecolor': 'black'}  # Error bars style
)
plt.title('Simmetric mean absolute percentage error in the prediction of duration')
plt.xlabel('Model')
plt.ylabel('SMAPE [%]')
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
plt.legend(title='Metrics')
plt.grid(axis='y', linestyle='--', linewidth=0.7)

plt.show()







column_errors_timestamp=[col for col in df.columns if col.startswith('Error @') or col.startswith('Error@')]
column_errors_erros=[col for col in df.columns if col.startswith('Error_error')]
data=df[column_errors_timestamp]
data_error=df[column_errors_erros]
# Plot the data
#a=data.plot(kind='bar', figsize=(10, 6))
a = data.plot(
    kind='bar',
    figsize=(10, 6),
    yerr=data_error.values.T,  # Transpose to match the shape
    capsize=5,                 # Add caps to the error bars
    error_kw={'elinewidth': 1, 'ecolor': 'black'}  # Error bars style
)
#plt.title('Error at different time points')
plt.xlabel('Model')
plt.ylabel('Time [seconds]')
a.set_yticks(list(a.get_yticks()) + [60, 120])

# Add custom labels for the new ticks
#a.text(-0.5, 60, '1min', color='red', ha='right', va='center', fontsize=12)
#a.text(-0.5, 120, '2min', color='red', ha='right', va='center', fontsize=12)
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
plt.grid(axis='y', linestyle='--', linewidth=0.7)

plt.show()




r'''
columns_mae = [col for col in df.columns if col.startswith('MAE')]
columns_smape = [col for col in df.columns if col.startswith('SMAPE')]
plot_mae = df[columns_mae]
plot_smape = df[columns_smape]

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
# Set the font sizes
plt.rcParams.update({
    'font.size': 14,         # General font size
    'axes.titlesize': 16,    # Title font size
    'axes.labelsize': 16,    # X and Y label font size
    'xtick.labelsize': 12,   # X-axis tick label font size
    'ytick.labelsize': 14,   # Y-axis tick label font size
    'legend.fontsize': 12,   # Legend font size
    'figure.titlesize': 18   # Figure title font size
})
# Plot the MAE data
plot_mae.plot(kind='bar', ax=axes[0])
axes[0].set_title('Mean Absolute Error in the Prediction of Duration')
#axes[0].set_xlabel('Model')
axes[0].set_ylabel('MAE [min]')
axes[0].set_xticks(range(len(df.index)))
axes[0].set_xticklabels(df.Model, rotation=45)
axes[0].legend(title='Metrics')

# Plot the SMAPE data
plot_smape.plot(kind='bar', ax=axes[1])
axes[1].set_title('Symmetric Mean Absolute Percentage Error in the Prediction of Duration')
axes[1].set_xlabel('Model')
axes[1].set_ylabel('SMAPE [%]')
axes[1].set_xticks(range(len(df.index)))
axes[1].set_xticklabels(df.Model, rotation=45)
axes[1].legend(title='Metrics')

# Adjust layout
plt.tight_layout()
plt.show()



# Select the columns to plot for training and testing time
columns_training_time = [col for col in df.columns if col.startswith('Training')]
columns_testing_time = [col for col in df.columns if col.startswith('Testing')]

plot_training_time = df[columns_training_time]/60
plot_testing_time = df[columns_testing_time]

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))

# Plot training time
plot_training_time.plot(kind='bar', ax=axes[0])
axes[0].set_title('1-fold training time for the models')
axes[0].set_xlabel('Model')
axes[0].set_ylabel('Time [min]')
axes[0].set_xticks(range(len(df.index)))
axes[0].set_xticklabels(df.Model, rotation=45)
axes[0].legend()

# Plot testing time
plot_testing_time.plot(kind='bar', ax=axes[1])
axes[1].set_title('1-fold testing time for the models')
axes[1].set_xlabel('Model')
axes[1].set_ylabel('Time [sec]')
axes[1].set_xticks(range(len(df.index)))
axes[1].set_xticklabels(df.Model, rotation=45)
axes[1].legend()

# Update general font size and other rcParams
plt.rcParams.update({
    'font.size': 14,         # General font size
    'axes.titlesize': 16,    # Title font size
    'axes.labelsize': 16,    # X and Y label font size
    'xtick.labelsize': 12,   # X-axis tick label font size
    'ytick.labelsize': 14,   # Y-axis tick label font size
    'legend.fontsize': 12,   # Legend font size
    'figure.titlesize': 18   # Figure title font size
})

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


'''


'''

data= df.iloc[:, -6:]
index_to_remove = data.shape[1] - 3

# Rimuoviamo la terz'ultima colonna usando il metodo .iloc
data = data.iloc[:, :index_to_remove].join(data.iloc[:, index_to_remove + 1:])
'''

