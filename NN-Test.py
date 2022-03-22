import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Aufbau des NN:
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2,10)
        self.fc2 = nn.Linear(10,10)
        self.fc3 = nn.Linear(10,1)

    def forward(self,x):
        x=x.to(torch.float32)
        x= F.relu(self.fc1(x))
        x= F.relu(self.fc2(x))
        x= torch.sigmoid(self.fc3(x))

        return x


#Datenverarbeitung:
data = np.loadtxt("data/no_colums_iris_binary.csv", delimiter=",")  # mit Header:,skiprows = 1
x,y = torch.from_numpy(data[:, 0:-3]), torch.from_numpy(data[:, -1:])
x = x.to(torch.float32)
y = y.to(torch.float32)
X_train, X_test, y_train, y_test =train_test_split(x,y, test_size=0.2, random_state=2)

#Definitionen
model = Model()
#loss = nn.CrossEntropyLoss()
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.01)
epochs = 10000
epoch_errors =[]
accuracy_list =[]
print (x.size())
print(y.size())
#Training
for epoch in range(epochs):
    error = loss(model(X_train),y_train)
    epoch_errors.append(error.item())
    error.backward()
    optimizer.step()

print(model(torch.tensor([3,2])))
def showTensor(aTensor):
    plt.figure()
    plt.imshow(aTensor.numpy())
    #plt.imshow(aTensor)
    plt.colorbar()
    plt.show()

x_plot=np.arange(0,10000)
print (len(epoch_errors))
print (len(x_plot))
plt.plot(x_plot,epoch_errors)
plt.show()
#print(epoch_errors)
