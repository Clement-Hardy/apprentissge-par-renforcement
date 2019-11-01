# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 02:09:22 2019

@author: Clement_X240
"""

from dqn import DQN

"""
The only thing changing from the DQN is the neural network using
Here, we need to modify the do_one_step function 
to add noise in the layers at each step
"""


class Noisy(DQN):
    
    def __init__(self, env, model, target_model, config, name_agent="Noisy-dqn"):
        super(Noisy, self).__init__(env, model, target_model, config, name_agent=name_agent)
        
    def do_one_step(self):
        
        loss = self.loss()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # adding noises in the layers of both neural networks.
        self.model.add_noise()
        self.target_model.add_noise()

        self.loss_data.append(loss.item())
