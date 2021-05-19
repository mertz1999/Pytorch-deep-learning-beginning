import numpy as np

# --- initializing
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)
W = 0.0

# --- model prediction
def forward(x):
    return W * x

# --- loss = MSR
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

# --- gradient
# ---- MSE = 1/N * (w*x - y)**2
# ---- dJ/dw = 1/N 2x (w*x - y) 
def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# --- Training
learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
    # --- Prediction
    y_pred = forward(X)

    # --- loss
    l = loss(Y, y_pred)

    # --- Gradient
    dw = gradient(X, Y, y_pred)

    # --- update weights
    W -= learning_rate * dw

    if epoch % 1 == 0:
        print(f'epoch  {epoch+1}: w = {W:.3f}, loss = {l:.8f}')

print("------------------------------------------")
print(f'Prediction After training: f(5) = {forward(5):.3f}')