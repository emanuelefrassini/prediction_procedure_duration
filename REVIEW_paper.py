# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:38:30 2024

@author: emanuelefrassi
"""

import numpy as np

# Load the .npz file
file_path = r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Codes\MTSC\time_binary_regression\data\dataset.npz"
data = np.load(file_path)
array_name = data.files[0]  # Replace with the actual name of the array
array_data = data[array_name]
# Check which numbers are present
all_numbers = np.arange(14)
present_numbers=[]
missing_numbers=[]
for i in range(array_data.shape[0]):
    present_numbers.append(np.intersect1d(array_data[i], all_numbers))

# Find missing numbers
    missing_numbers.append(np.setdiff1d(all_numbers, array_data[i]))
phase_fa=[]
phase_ha=[]
for i in range(array_data.shape[0]):
    if len(np.where(array_data[i]==6)[0])>0:
        phase_fa.append(i)
    if len(np.where(array_data[i]==8)[0])>0:
        phase_ha.append(i)

target_index = 2944
tolerance = 1  # Define "close" as within 5 of 49

# Find the indices
indices = []
duration = []

for i in range(array_data.shape[0]):  # Iterate through the first axis
    flattened_slice = array_data[i].flatten()  # Flatten the 2D slice for easy search
    try:
        first_occurrence = np.where(flattened_slice == -1)[0][0]  # Find the first occurrence of -1
        if abs(first_occurrence - target_index) <= tolerance:  # Check if it meets the condition
            indices.append(i)
            duration.append(first_occurrence)
    except IndexError:
        # If there is no -1, skip this slice
        continue

print("Indices on the first axis that meet the condition:", indices)
data=array_data[19]
unique_values, start_indices = np.unique(data, return_index=True)

# Find the end indices of each consecutive group
end_indices = np.append(start_indices[1:], len(data)) - 1

# Combine the results into a list of tuples
result = [(value, start, end) for value, start, end in zip(unique_values, start_indices, end_indices)]

# Print the results
for value, start, end in result:
    print(f"Value: {value}, Start Index: {start/60}, End Index: {end/60}")