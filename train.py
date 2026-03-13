import numpy as np

#load data 
print("Loading data...")
x = np.load('data/x_train.npy') 
y = np.load('data/y_train.npy')

# !!! data normalization
x = x / np.max(x) # scale features to [0, 1] range to improve training stability and convergence speed

# m is the total number of battles (1000)
m = x.shape[1]

# network architecture 
input_size = 12
hidden_size = 8
output_size = 1

# initialization of weights and biases
#initialize weights randomly to break symmetry but keep them small to prevent saturation of activation functions
np.random.seed(42) # for reproducibility

#layer 1 (hidden layer)
w1 = np.random.randn(hidden_size, input_size) * 0.01 # small random weights
b1 = np.zeros((hidden_size, 1)) # zero biases

#layer 2 (output layer)
w2 = np.random.randn(output_size, hidden_size) * 0.01 # small random weights
b2 = np.zeros((output_size, 1)) # zero biases

# activation function (ReLU for hidden layer, sigmoid for output layer)
def relu(z):
    """used in hidden layer to learn non-linear patterns"""
    return np.maximum(0, z)

def sigmoid(z):
    """used in output layer for binary classification to output probabilities"""
    return 1 / (1 + np.exp(-z))

print ("Network architecture and parameters initialized.")

# hyperparameters
epochs = 5000 # number of times to iterate over the entire training dataset
learning_rate = 0.05 # step size for updating weights and biases during training, higher learning rate can speed up training but may overshoot minima, lower learning rate can lead to more stable convergence but may take longer to train

print ("\nStarting Training Loop...\n")

#training loop

for epoch in range(epochs):
    #forward pass

    #hidden layer
    z1 = np.dot(w1, x) + b1 # linear transformation
    a1 = relu(z1) # activation
    #output layer
    z2 = np.dot(w2, a1) + b2 # linear transformation
    a2 = sigmoid(z2) # activation

    #calculate the loss (binary cross-entropy)
    # math : L = -[y* log(a2) + (1 - y) * log(1 - a2)] averaged over all examples
    # add a small number 1e-8 to prevent log(0) which is undefined
    loss = - (1/m) * np.sum(y * np.log(a2 + 1e-8) + (1 - y) * np.log(1 - a2 + 1e-8))

    #backpropagation
    # output layer error
    dz2 = a2 - y # derivative of loss with respect to z2

    #calculate gradients for w2 and b2
    dw2 = (1/m) * np.dot(dz2, a1.T)
    db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)

    #push error back to hidden layer
    dz1 = np.dot(w2.T, dz2) * (z1 > 0) # derivative of ReLU

    #calculate gradients for w1 and b1
    dw1 = (1/m) * np.dot(dz1, x.T)
    db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)

    #update weights and biases
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2

    #logging every 500 epochs
    if epoch % 500 == 0:
        predictions = (a2 > 0.5).astype(int)
        accuracy = np.mean(predictions == y) * 100
        print(f'Epoch {epoch:4d}: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}%')

print("\nTraining complete.")

# save the trained parameters for later use in prediction

print("\nSaving trained parameters...")

# save to data folder
np.save('data/w1.npy', w1)
np.save('data/b1.npy', b1)
np.save('data/w2.npy', w2)
np.save('data/b2.npy', b2)

print("Parameters saved successfully.")