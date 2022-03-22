import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import optuna

DEVICE= "cuda:0"
def define_model(trial):
    # We optimize the number of layers, hidden untis and dropout ratio in each layer.
    n_layers = trial.suggest_int("layers{}".format(i), 0.2, 0.5)
    layers = []

    in_features = 2
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_uniform("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)

def objective(trial):

    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name =  optuna.Trial.suggest_cat("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = optuna.Trail.suggest_loguniform("lr", 1e-5, 1e-1)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    return accuracy

nlayers= optuna.Trial.suggest_int("layers_l", 0.2, 0.5)
nlayers=[]
nfeatures= 2
layers=[]
for i in range(nlayers):

    hiddensize = optuna.Trial.suggest_int("n_units_1{}".format(i),4,128)
    layers.append(nn.Linear(nfeatures,hiddensize))
    layers.append(nn.Dropout(dropout))
    layers.append(nn.ReLU)
    p = optuna.Trial.suggest_uniform("dropout:{}",format(i),0.2,0.5)
    layers.append(nn.Dropout(p))
    layers.append(nn.Linear(hiddensize,ntargets))
    self.model = nn.Sequential(*layers)

    def forward(self,x):
        return self.model(x)

def objective(trail):
    params ={
        "num_layers": optuna.trial.sugg
    }

#Datenverarbeitung:
data = np.loadtxt("data/no_colums_iris_binary.csv", delimiter=",")  # mit Header:,skiprows = 1
x,y = torch.from_numpy(data[:, 0:-3]), torch.from_numpy(data[:, -1:])
x = x.to(torch.float32)
y = y.to(torch.float32)
X_train, X_test, y_train, y_test =train_test_split(x,y, test_size=0.2, random_state=2)

#Definitionen

nfeatures = X_train.shape[1]
ntargets = y_train.shape[1]

hiddensize = 5
dropout = 0.2
model = Model(ntargets,nfeatures, nlayers,hiddensize,dropout)

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epochs = 1000
epoch_errors =[]
accuracy_list =[]
print (x.size())
print(y.size())
#Training
for epoch in range(epochs):
    error = loss(model(X_train),y_train)
    epoch_errors.append(error.item)
    error.backward()
    optimizer.step()

print(model(torch.tensor([3,2])))
