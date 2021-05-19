import torch

x = torch.randn(3, requires_grad=True)

y = x+2

z = y.mean()

z.backward() # --- dy/dx


#  --- Stop Grad 3 methods
# x.requires_grad_(False)
# y = x.detach()
# with torch.no_grad():

# --- training
weights = torch.randn(3, requires_grad=True)
for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    # weights.grad.zero_()