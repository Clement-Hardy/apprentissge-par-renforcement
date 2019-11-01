# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 02:06:28 2019

@author: Clement_X240
"""
import numpy as np
import torch
import torch.optim as optim
from dqn import DQN
from buffer import PrioritizedReplayBuffer


class Prioritized(DQN):
    
    def __init__(self, env, model, target_model, config, name_agent="prioritized-dqn"):
        self.name_agent = name_agent
        
        self.dim_space = env.observation_space.shape[0]
        self.nb_actions = env.action_space.n
        
        self.epsilon = config.epsilon_start
        self.epsilon_final = config.epsilon_final
        self.epsilon_start = config.epsilon_start
        self.epsilon_decay = config.epsilon_decay
        
        self.gamma = config.gamma
        self.update_nb_iter = config.update_nb_iter
        
        # changing the buffer (taking a priotirized buffer
        # insted of a uniform probability buffer)
        self.replay_buffer = PrioritizedReplayBuffer(10000, config.batch_size, config.w,
                                                     config.beta_final, config.beta_start, config.beta_decay)
        self.environment = env
        self.batch_size = config.batch_size

        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        
        self.loss_data = []
        self.rewards = []            
        
    def loss(self):
        """ 
            the loss is equal to:
                    Rt+1+γt+1qθ(St+1,argmax qθ(St+1,a′))−qθ(St,At))^2
        """
        states, actions, rewards, next_states, finish, indices, weight = self.replay_buffer.sample()
        actions = actions.long()
        
        # qθ(St,At)
        q0 = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # argmax qθ_barre(St+1,a′)
        max_next_q0 = self.model(next_states).max(1)[0] *(1-finish)
        
        Rt_gamma_max = (rewards + self.gamma * max_next_q0)
        
        loss = (q0 - Rt_gamma_max).pow(2) * weight
        
        # update the priority of the buffer
        self.replay_buffer.add_p(indices, loss.detach().numpy())
        
        loss = loss.sum()
        
        return loss
