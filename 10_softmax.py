import torch 

x = torch.tensor([2.0, 1.0, 0.1])
result = torch.softmax(x, dim=0)
print(result)