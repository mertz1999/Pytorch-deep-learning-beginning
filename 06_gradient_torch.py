# 1) Design model (input, output, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#    - forward pass
#    - bachward pass
#    - update weights

import torch
import torch.nn as nn


# --- initializing
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
print(f'number of samples: {n_samples} and number of features: {n_features}')

# --- model
input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)

# --- test
print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# --- Training
learning_rate = 0.01
n_iters = 10

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # --- Prediction
    y_pred = model(X)

    # --- loss
    l = loss(Y, y_pred)

    # --- Gradient
    l.backward() # dl/dw

    # --- update weights
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 1 == 0:
        [W, b] = model.parameters()
        print(f'epoch {epoch+1}: W = {W[0][0].item():.3f}, loss = {l.item():.8f}')

print("------------------------------------------")
print(f'Prediction After training: f(5) = {model(X_test).item():.3f}')