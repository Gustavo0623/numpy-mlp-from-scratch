import numpy as np

print ("Loading data...")

# load thge optimized weights and biases from training
w1 = np.load('data/w1.npy')
b1 = np.load('data/b1.npy')
w2 = np.load('data/w2.npy')
b2 = np.load('data/b2.npy')

# define activation functions

def sigmoid(z):
    """used in output layer for binary classification to output probabilities"""
    return 1 / (1 + np.exp(-z))

def relu(z):
    """used in hidden layer to learn non-linear patterns"""
    return np.maximum(0, z)

# prediction engine
def predict_battle(stats_a, stats_b):
    # combine the stats of both Pokemon into a single 12 column vector
    x_new = np.array(stats_a + stats_b).reshape(12, 1)

    # normalization (same as training)
    # since max stat is 150, divide by 150
    x_new = x_new / 150.0

    # forward pass through the network
    z1 = np.dot(w1, x_new) + b1
    a1 = relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    # interpret ouput probability
    
    probability = a2[0, 0] # extract scalar from array
    if probability > 0.5:
        print(f"Pokemon A is predicted to win with probability {probability * 100:.2f}")
    else:
        print(f"Pokemon B is predicted to win with probability {(1 - probability) * 100:.2f}")

# test
# stats order is: HP, Attack, Defense, Sp. Atk, Sp. Def, Speed for each Pokemon

# pikachu (fast but fragile)
pikachu_stats = [35, 55, 40, 50, 50, 90]

# snorlax (massive hp and attack but slow)
snorlax_stats = [160, 110, 65, 65, 110, 30]

print ("Predicting battle outcome between Pikachu and Snorlax...")
predict_battle(pikachu_stats, snorlax_stats)

# mewtwo overpowered stats for testing
mewtwo = [106, 110, 90, 154, 90, 130]

# caterpie weak stats for testing
caterpie = [45, 30, 35, 20, 20, 45]

print("Predicting battle outcome between Mewtwo and Caterpie...")
predict_battle(mewtwo, caterpie)