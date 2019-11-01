# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 13:24:38 2019

@author: Clement_X240
"""
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

"""
create a noisy layer
two method: 
    Factorized; use p+q unit gaussian variables
    not Factorized: use pq+q gaussian variables
    
    p: inputs size
    q: outputs size
"""


class Noisy(nn.Module):
    def __init__(self, input_shape, output_shape, sigma=0.5, factorized=True):
            super(Noisy, self).__init__()
            
            self.dim_space = input_shape
            self.nb_actions = output_shape
            self.sigma = sigma # std of gaussian variables
            self.Factorized = factorized
            
            self.epsilon_weights = torch.empty(self.nb_actions, self.dim_space)
            self.epsilon_b = torch.empty(self.nb_actions)
            
            # initialized the parameters of the gaussian
            self.init_parameters()
            
            # add gaussian noise
            self.add_noise()

    def init_parameters(self):
        """ 
         initialized the parameters of the gaussian
         
         Two cases: Factorized, not Factorized
         
         p: input size
         sigma: std of gaussian pass in argument (when created the layer)
         
         not Factorized:
                 u_ij ~ uniform[-sqrt(3/p), sqrt(3/p)]
                 sigma_ij = 0.017
                 
        Factorized:
                u_ij ~ uniform[-1/sqrt(p), 1/sqrt(p)]
                 sigma_ij = sigma/sqrt(p)                 
         """ 
        
        if self.Factorized:
            borne = 1. / np.sqrt(self.dim_space)
            self.u_weights = torch.FloatTensor(self.nb_actions, self.dim_space).uniform_(-borne, borne)
            value = self.sigma / np.sqrt(self.dim_space)
            self.sigma_weigths = torch.FloatTensor(self.nb_actions, self.dim_space).fill_(value)
            self.u_b = torch.FloatTensor(self.nb_actions).uniform_(-borne, borne)
            self.sigma_b = torch.FloatTensor(self.nb_actions).fill_(value)
        
        else:
            borne = np.sqrt(3. / self.dim_space)
            self.u_weights = torch.FloatTensor(self.nb_actions, self.dim_space).uniform_(-borne, borne)
            self.sigma_weigths = torch.FloatTensor(self.nb_actions, self.dim_space).fill_(self.sigma)
            
            self.u_b = torch.FloatTensor(self.nb_actions).uniform_(-borne, borne)
            self.sigma_b = torch.FloatTensor(self.nb_actions).fill_(self.sigma)
        
        self.u_weights = nn.Parameter(self.u_weights)
        self.sigma_weigths = nn.Parameter(self.sigma_weigths)
        self.u_b = nn.Parameter(self.u_b)
        self.sigma_b = nn.Parameter(self.sigma_b)

    @staticmethod
    def f(x):
        """
            for the factorized case:
                the noises are factorized like:
                    ε^w_(i,j)=f(ε_i)f(ε_j)
                    ε^b_j=f(ε_j)
            
            we will use f(x)=sign(x)*sqrt(|x|)
        """
        return x.sign().mul_(x.abs().sqrt_())
    
    def add_noise(self):

        if self.Factorized:
            x = torch.distributions.Normal(loc=0, scale=1)
            y = x.sample((self.dim_space,))
            epsilon_i = torch.Tensor(self.f(y))
        
            y = x.sample((self.nb_actions,))
            epsilon_j = torch.Tensor(self.f(y))
            self.epsilon_weights.copy_(epsilon_j.ger(epsilon_i))
            self.epsilon_b.copy_(epsilon_j)
        
        else:
            x = torch.distributions.Normal(loc=0, scale=1)
            self.epsilon_weights= x.sample((self.nb_actions, self.dim_space))
            self.epsilon_b = x.sample((self.nb_actions,))
            
    def forward(self, y):
        """
            we need to modify the forward function of the layer as it's not:
                f(y) = f(wx + b) anymore
        """
        if self.training:
            return F.linear(y, self.u_weights + self.sigma_weigths * self.epsilon_weights,
                            self.u_b + self.sigma_b * self.epsilon_b)
        else:
            return F.linear(y, self.u_weights, self.u_b)
