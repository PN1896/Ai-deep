import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

#Aufbau des NN:
class Model(torch.nn.Module):
    def __init__(self):
        super(mopdel, self).__init__()
        self.fc1 = nn.Linear(2,10)
        self.fc2 = nn.Linear(10,10)
        self.fc3 = nn.Linear(10,1)

    def forward(self,x):
        x=x,to(torch.float32)
        x= F.relu(self.fc1(x))
        x= F.relu(self.fc2(x))
        x= torch.sigmoid(self.fc3(x))

        return x

#Datenverarbeitung:
data = np.loadtxt ("./data/no_colums_iris_binary.csv", delimiter=",")
