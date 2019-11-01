# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 02:37:30 2019

@author: Clement_X240
"""


class Config(object):
    
    gamma = 0.99            # discounting factor
    batch_size = 32
    learning_rate = 1e-4
    update_nb_iter = 200    # frequency to update the target model from the online network
    
    epsilon_final = 0.01    # minimum exploration probability
    epsilon_start = 0.99    # exploration probability at start
    epsilon_decay = 200     # exponential decay rate for exploration prob

    beta_start = 0.4        # importance sampling parameter
    beta_final = 1.         # annealed up to 1 during training
    beta_decay = 200        # incrementation per sampling
    
    w = 0.8             # determines the shape of the distribution of the probability to be sampled of past experiences

    nb_steps = 3            # multi-step learning parameter
    
    factorized = True       # Gaussian noise can be factorized or independent
    sigma = 0.5             # variance of the Gaussian noise in Noisy Nets

    # Distributional RL parameters
    nb_atoms = 51
    Z_min = -10
    Z_max = 10
