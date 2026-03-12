import numpy as np

np.random.seed(42)

num_battles = 1000

print(f"Genrating {num_battles} simulated Pokemon battles...")

# simulate stats ranging from 30-150
pokemon_a_stats = np.random.randint(30, 150, size=(num_battles, 6))
pokemon_b_stats = np.random.randint(30, 150, size=(num_battles, 6))

# combine them into massive input matrix (X), stacks side-side giving 12 columns per row
X_train = np.hstack((pokemon_a_stats, pokemon_b_stats))

# determine winner to create labels (Y)
total_a = np.sum(pokemon_a_stats, axis=1)
total_b = np.sum(pokemon_b_stats, axis=1)

# Rule : higher total stats wins but we add rng luck to the battle. Y will be 1 if A wins and 0 if B wins
Y_train = (total_a + np.random.randint(-40, 40, size=num_battles) > total_b).astype(int)

#reshape y to be a tall column vecxtor so the matrix math aligns with neural network
Y_train = Y_train.reshape(1, num_battles) #shape (1, 1000)

#transpose x so that math aligns with formulas
X_train = X_train.T #shape (12, 1000)

np.save('data/X_train.npy', X_train)
np.save('data/Y_train.npy', Y_train)

print(f"Success! Saved X_train.npy to the data/ folder. Shape: {X_train.shape}")
print(f"Success! Saved Y_train.npy to the data/ folder. Shape: {Y_train.shape}")