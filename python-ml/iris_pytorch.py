#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #Setup neural net (4 => 100 => 100 => 3) nodes
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 3)
        #Squash tensor elements to range of 0,1
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.softmax(X)

        return X

#Load data
url = 'C:/dev/ml-prototyping/python-ml/iris-data.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']
dataset = pandas.read_csv(url, names=names)

#Transform species class to numeric
dataset.loc[dataset.species=='Iris-setosa', 'species'] = 0
dataset.loc[dataset.species=='Iris-versicolor', 'species'] = 1
dataset.loc[dataset.species=='Iris-virginica', 'species'] = 2

#Divide data into test/train data/species
data = dataset.values[:,0:4]
species = dataset.values[:,4]
train_data, test_data, train_species, test_species = train_test_split(data, species, test_size=0.8)

#Put data/species into tensors
train_data = Variable(torch.Tensor(train_data).float())
test_data = Variable(torch.Tensor(test_data).float())
train_species = Variable(torch.Tensor(train_species).long())
test_species = Variable(torch.Tensor(test_species).long())

#Initialize neural net
net = Net()
#Initialize loss function
criterion = nn.NLLLoss()
#Initialize optimizer function
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

#Training loop
for epoch in range(10000):
    #Clear optimizer
    optimizer.zero_grad()
    #Feed training data to neural net
    out = net(train_data)
    #Get loss (incorrect species) amount
    loss = criterion(out, train_species)
    #Propigate loss for next loop
    loss.backward()
    optimizer.step() 

    if epoch % 100 == 0:
        print('number of epoch', epoch, 'loss', loss.data.item())

#Test the neural net
predict_out = net(test_data)
_, predict_species = torch.max(predict_out, 1)

#Accuracy
print('prediction accuracy', accuracy_score(test_species.data, predict_species.data))
#Precision, ratio: true positives / (true positives + false positives)
print('micro precision', precision_score(test_species.data, predict_species.data, average='micro'))
#Recall, ration: true positives / (true positives + false negatives)
print('micro recall', recall_score(test_species.data, predict_species.data, average='micro'))

#Single prediction
single_predict = [[
    5.5,
    2.4,
    3.8, 
    1.1
]]
predict_out = net(Variable(torch.Tensor(single_predict)))
_, predict_species = torch.max(predict_out, 1)
species = predict_species.data.item()
print()
print('One-off prediction', 'Iris-setosa' if species == 0 else 'Iris-versicolor' if species == 1 else 'Iris-virginica' if species == 2 else 'unknown')