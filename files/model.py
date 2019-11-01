# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 02:28:39 2019

@author: Clement_X240
"""
import torch.nn as nn
import torch.nn.functional as F
from layer import Noisy


def model_basic(input_shape, output_shape):
    model = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_shape)
        )
    return model

        
class ModelDuelings(nn.Module):
    
    def __init__(self, input_shape, output_shape):
        super(ModelDuelings, self).__init__()
        
        self.dim_space = input_shape
        self.nb_actions = output_shape
        
        self.fc1 = nn.Linear(self.dim_space, 64)
        
        self.advantage1 = nn.Linear(64, 64)
        self.advantage2 = nn.Linear(64, self.nb_actions)
        
        self.value1 = nn.Linear(64, 64)
        self.value2 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        
        ad = self.advantage1(x)
        ad = F.relu(ad)
        ad = self.advantage2(ad)
        
        va = self.value1(x)
        va = F.relu(va)
        va = self.value2(va)
        
        return va + ad - ad.mean()
    

class ModelDistributional(nn.Module):
    
    def __init__(self, input_shape, output_shape, nb_atoms=51):
        super(ModelDistributional, self).__init__()
        
        self.dim_space = input_shape
        self.nb_actions = output_shape
        self.nb_atoms = nb_atoms
        
        self.fc1 = nn.Linear(self.dim_space, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, self.nb_actions * self.nb_atoms)
        
    def forward(self, x, log=False):
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        
        if log:
            x = F.log_softmax(x.view(-1, self.nb_atoms)).view(-1, self.nb_actions, self.nb_atoms)
        else:
            x = F.softmax(x.view(-1, self.nb_atoms)).view(-1, self.nb_actions, self.nb_atoms)
        
        return x
    

class ModelRainbow(nn.Module):
    
    def __init__(self, input_shape, output_shape, sigma=0.5, factorized=True, nb_atoms=51):
        super(ModelRainbow, self).__init__()
        
        self.dim_space = input_shape
        self.nb_actions = output_shape
        self.nb_atoms = nb_atoms
        self.factorized = factorized
        
        self.fc1 = nn.Linear(self.dim_space, 64)
        
        self.advantage1 = Noisy(64, 64, sigma, factorized)
        self.advantage2 = Noisy(64, self.nb_actions * self.nb_atoms, sigma, factorized)
        
        self.value1 = Noisy(64, 64, sigma, factorized)
        self.value2 = Noisy(64, self.nb_atoms, sigma, factorized)
        
    def forward(self, x, log=False):
        x = self.fc1(x)
        x = F.relu(x)
        
        ad = self.advantage1(x)
        ad = F.relu(ad)
        ad = self.advantage2(ad).view(-1, self.nb_actions, self.nb_atoms)
        
        va = self.value1(x)
        va = F.relu(va)
        va = self.value2(va).view(-1, 1, self.nb_atoms)
        if log:
            return F.log_softmax(va + ad - ad.mean(1).view(-1, 1, self.nb_atoms), 2)
        else:
            return F.softmax(va + ad - ad.mean(1).view(-1, 1, self.nb_atoms), 2)
        
        
class ModelNoisy(nn.Module):
    
    def __init__(self, input_shape, output_shape, factorized=True, sigma = 0.5):
        super(ModelNoisy, self).__init__()
        
        self.dim_space = input_shape
        self.nb_actions = output_shape
        self.sigma = sigma
        
        self.fc1 = Noisy(self.dim_space, 64, sigma, factorized)
        self.fc2 = Noisy(64, 64, sigma, factorized)
        self.fc3 = Noisy(64, self.nb_actions, sigma, factorized)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return x
    
    def add_noise(self):
        
        self.fc1.add_noise()
        self.fc2.add_noise()
        self.fc3.add_noise()
