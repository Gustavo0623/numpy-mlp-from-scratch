import torch
import numpy as np

print ("initializing pytorch...")

# hardware acceleration (GPU)

#pytorch can dynamically use the GPU if available, otherwise it will use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (f"target compute device: {device.type.upper()}")

# load and translate data

print ("loading historical numpy matrices...")
x_np = np.load('data/x_train.npy')
y_np = np.load('data/y_train.npy')

# !! pytorch and numpy prefer different matrix shapes
# numpy mlp features are rows, battles are columns (12, 1000)
# pytorch mlp features are columns, battles are rows (1000, 12)
x_np = x_np.T
y_np = y_np.T

# normalize data
x_np = x_np / np.max(x_np)

print ("converting to pytorch tensors and routing to device")
x = torch.tensor(x_np, dtype=torch.float32).to(device)
y = torch.tensor(y_np, dtype=torch.float32).to(device)

# verification
print (f"data shapes: x={x.shape} | location: {x.device}, y={y.shape} | location: {y.device}")
print ("data pipeline initialized successfully")