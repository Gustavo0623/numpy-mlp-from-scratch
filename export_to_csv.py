import numpy as np
import pandas as pd

print("Loading binary matrices...")
# Load the fast, machine-readable data
X = np.load('data/X_train.npy')
Y = np.load('data/Y_train.npy')

# X is currently (12, 1000) for our math. 
# We need to flip it back to (1000, 12) so each row is one battle.
X_flat = X.T 
Y_flat = Y.T

# Stack the labels (Y) onto the end of the features (X)
# This creates a single matrix with 13 columns (12 stats + 1 winner)
combined_data = np.hstack((X_flat, Y_flat))

# Define the human-readable column names
columns = [
    'Pokemon_A_HP', 'Pokemon_A_Atk', 'Pokemon_A_Def', 'Pokemon_A_SpA', 'Pokemon_A_SpD', 'Pokemon_A_Spe',
    'Pokemon_B_HP', 'Pokemon_B_Atk', 'Pokemon_B_Def', 'Pokemon_B_SpA', 'Pokemon_B_SpD', 'Pokemon_B_Spe',
    'Winner_Is_A'
]

print("Converting to human-readable CSV format...")
# Create a pandas DataFrame and save it as a CSV
df = pd.DataFrame(combined_data, columns=columns)
df.to_csv('data/synthetic_pokemon_battles.csv', index=False)

print("Success! 'synthetic_pokemon_battles.csv' is ready to upload to Kaggle.")