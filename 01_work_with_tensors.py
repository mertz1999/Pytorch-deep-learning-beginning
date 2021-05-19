import torch
import numpy as np

# --- Data
x = torch.empty(1)
y = torch.rand(1)
z = torch.tensor([12, 22, 11])

# --- Operator
a = torch.rand(2, 2)
b = torch.rand(2, 2)
b.add_(a)   # mul_ sub_ div_ 


# --- Slicing
a = torch.rand(5, 3)
# print(a[:,1])

# --- reshape
a = torch.rand(4, 4)
b = a.view(16)
c = a.view(-1, 8)

# --- to numpy
a = torch.ones(2, 2)
b = a.numpy()
a.add_(1)

# --- to tensor
a = np.ones(2)
b = torch.from_numpy(a)
a += 1

