import torch 
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import math 


class WineDataset(Dataset):
    def __init__(self):
        # data Loading
        xy = np.loadtxt('./data/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:,1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

# --- Go to Training loop
total_sample = len(dataset)
n_iterations = math.ceil(total_sample/4)
n_epoch = 2

for epoch in range(n_epoch):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward
        if (i+1)%5 == 0:
            print(f'epoch {epoch+1}/{n_epoch} , step {i+1}/{n_iterations} , input {inputs.shape}')


