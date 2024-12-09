# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 12:34:40 2024

@author: emanuelefrassi
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
# Load the dataset
file_path = r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Datasets\Haga\Dataset_fully_anonomyzed.xlsx"
df = pd.read_excel(file_path)

# Filter out rows where 'besteld' is zero
filtered_df = df[df['besteld'] != 0]
filtered_df=filtered_df[filtered_df['gerealiseerde procedure']=='CAG']

filtered_df['actual_besteld']=filtered_df['besteld']-filtered_df['besteld']
filtered_df['actual_besteld'] = np.where(
    filtered_df['Aankomst OK'] != 0,
    filtered_df['Aankomst OK'] - filtered_df['besteld'],
    np.where(
        filtered_df['Start inleiding'] != 0,
        filtered_df['Start inleiding'] - filtered_df['besteld'],
        filtered_df['Start operateur(CAR)'] - filtered_df['besteld']
    )
)
filtered_df = filtered_df[filtered_df['actual_besteld'] >= 0]
# Compute mean and standard deviation
mean_besteld = filtered_df['actual_besteld'].mean()
std_besteld = filtered_df['actual_besteld'].std()

# Print the results
print(f"Mean of 'besteld': {mean_besteld}")
print(f"Standard Deviation of 'besteld': {std_besteld}")

# Plot the distribution
plt.figure(figsize=(10, 6))
sns.histplot(filtered_df['actual_besteld'], kde=True)
plt.title("Distribution of 'besteld'")
plt.axvline(mean_besteld, color='red', linestyle='--', label=f'Average: {mean_besteld:.2f}')
plt.text(mean_besteld, -12, f'{mean_besteld:.2f}', 
         color='red', ha='center', va='center', backgroundcolor='white')
plt.xlabel('besteld')
plt.ylabel('Frequency')
plt.show()
save_path = r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Haga\Data analysis\besteld_time.png"

plt.savefig(save_path)

procedure_counts = df['gerealiseerde procedure'].value_counts()

plt.figure(figsize=(10, 6))
plt.pie(procedure_counts, labels=procedure_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Procedure Types')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
save_path = r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Haga\Data analysis\Procedures_count.png"

plt.savefig(save_path)

df['CAR_difference'] = df['Eind operateur(CAR)'] - df['Start operateur(CAR)']

# Get unique procedures
unique_procedures = df['gerealiseerde procedure'].unique()

# Plot distributions
for procedure in unique_procedures:
    subset = df[df['gerealiseerde procedure'] == procedure]
    
    missing_values_count = ((subset['Start operateur(CAR)'] == 0) | (subset['Eind operateur(CAR)'] == 0)).sum()

    print(f"Procedure {procedure}: Total number = {len(subset)}, Missing values = {missing_values_count}, Percentage of NA = {missing_values_count/len(subset)*100}%\n")

    
    plt.figure(figsize=(10, 6))
    sns.histplot(subset['CAR_difference'], kde=True)
    plt.title(f'Procedure length: {procedure}')
    plt.xlabel('CAR Difference')
    plt.ylabel('Frequency')
    plt.show()
    title = f'Procedure length: {procedure}'
    filename = re.sub(r'[^\w\s-]', '', title).replace(' ', '_') + '.png'
    save_path = fr"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Haga\Data analysis\{filename}"
    plt.savefig(save_path)
  