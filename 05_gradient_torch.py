import torch

# --- initializing
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
W = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)

# --- model prediction
def forward(x):
    return W * x

# --- loss = MSR
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

print(f'Prediction before training: f(5) = {forward(5).item():.3f}')

# --- Training
learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
    # --- Prediction
    y_pred = forward(X)

    # --- loss
    l = loss(Y, y_pred)

    # --- Gradient
    l.backward() # dl/dw


    # --- update weights
    with torch.no_grad():
        W -= learning_rate * W.grad
    W.grad.zero_()

    if epoch % 4 == 0:
        print(f'epoch  {epoch+1}: w = {W.item():.3f}, loss = {l.item():.8f}')

print("------------------------------------------")
print(f'Prediction After training: f(5) = {forward(5).item():.3f}')