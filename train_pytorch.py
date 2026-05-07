import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print ("initializing pytorch engine...")

# hardware acceleration (GPU)

#pytorch can dynamically use the GPU if available, otherwise it will use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (f"target compute device: {device.type.upper()}")

# load and translate data

print ("loading historical numpy matrices...")
x_np = np.load('data/x_train.npy').T
y_np = np.load('data/y_train.npy').T

# normalize data
x_np = x_np / np.max(x_np)

# convert to torch tensors and route to gpu
x = torch.tensor(x_np, dtype=torch.float32).to(device)
y = torch.tensor(y_np, dtype=torch.float32).to(device)

# build the neural network

class PokemonPredictor(nn.Module):
    def __init__(self):
        super(PokemonPredictor, self).__init__()
        self.layer1 = nn.Linear(12, 8)    # hidden layer to output layer
        self.relu = nn.ReLU()           # activation function
        self.layer2 = nn.Linear(8,1)     # hidden layer to output layer
        self.sigmoid = nn.Sigmoid()         # output activation for binary classification

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.sigmoid(out)
        return out
    
# initialize the model and route to gpu
model = PokemonPredictor().to(device)

# pytorch built in binary cross entropy loss function
criterion = nn.BCELoss()

# stochastic gradient descent optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)

# the training loop
print
epochs = 5000

for epoch in range(epochs):
    # clear the gradients
    optimizer.zero_grad()

    # forward pass
    outputs = model(x)

    # compute the loss
    loss = criterion(outputs, y)

    # backward pass and optimize
    loss.backward()

    # update the weights
    optimizer.step()

    # print the progress every 500 epochs
    if epoch % 500 == 0 or epoch == epochs - 1:
        # calculate accuracy on the gpu
        predicted = (outputs > 0.5).float()  # convert probabilities to binary predictions
        accuracy = (predicted == y).float().mean()  # calculate accuracy
        print (f"Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.4f} | Accuracy: {accuracy.item():.2f}%")

print ("training complete!")