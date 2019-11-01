# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 02:11:52 2019

@author: Clement_X240
"""
from dqn import DQN
import torch
import numpy as np


class Distributional(DQN):
    
    def __init__(self, env, model, target_model, config, name_agent="Distributional-dqn"):
        
        super(Distributional, self).__init__(env, model, target_model, config, name_agent = name_agent)
                
        # discrete support (for the approximating distribution d_t)
        self.Z = torch.linspace(config.Z_min, config.Z_max, config.nb_atoms)
        self.Z_min = config.Z_min
        self.Z_max = config.Z_max
        self.nb_atoms = config.nb_atoms

    def action(self, state):
        """
            Select an action to take with an epsilon greedy strategy.

            Params
            ------
            state: 
                current position

            Returns
            -------
            action: int
                The action the agent will take.
        """
        
        # Exploration
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.nb_actions)
        
        # Exploitation
        else:
            prob_action = self.model(state) * self.Z
            action = prob_action.sum(2).argmax(1).item()
            return action

    def loss(self):
        """
            the loss is equal to:
                DKL(Φzd't||dt)
                
                with d't ≡ (R_t+1 + γ_t+1 z, pθ(S_t+1, a∗t+1))
        
        """
        states, actions, rewards, next_states, finish = self.replay_buffer.sample()
        actions = actions.long()
        
        # Q(xt+1,a) := somme_i(zi pi(xt+1,a))
        next_proba = (self.target_model(next_states) * self.Z)
        # print(next_proba.size())
        # a* = argmax_a(Q(xt+1,a))
        max_next_q0 = next_proba.sum(2).argmax(1).view(-1, 1, 1).expand(self.batch_size, -1, self.nb_atoms)
        
        # redimission the rewards, finish and the support to avoid the loop
        rewards = rewards.unsqueeze(1).expand(-1, self.nb_atoms)
        finish = finish.unsqueeze(1).expand(-1, self.nb_atoms)
        Z = self.Z.unsqueeze(0).expand(self.batch_size, -1)
        
        # compute the projection in the support of the support (by using clamp function)
        Tz = (rewards + (1-finish) * self.gamma * Z).clamp(min=self.Z_min, max=self.Z_max)
        b = (Tz - self.Z_min)/((self.Z_max - self.Z_min) / (self.nb_atoms - 1))
        l = b.floor().long()
        u = b.ceil().long()

        temp = torch.linspace(0, (self.batch_size - 1) * self.nb_atoms, self.batch_size).long().unsqueeze(1).expand(self.batch_size, self.nb_atoms)
        index_l = (temp + l).view(-1)
        index_u = (temp + u).view(-1)
        
        # p(xt+1,a∗)
        next_proba_action = next_proba.gather(1, max_next_q0).squeeze(1)
        
        # p(xt+1,a∗)(u−bj)
        data1 = (next_proba_action * (u.float() - b)).view(-1)
        
        # p(xt+1,a∗)(bj−l)
        data2 = (next_proba_action * (b - l.float())).view(-1)
        
        # the function index_add permit to do m=m + ... without using a loop
        m = torch.zeros(self.batch_size, self.nb_atoms)
        m.view(-1).index_add_(0, index_l, data1)  
        m.view(-1).index_add_(0, index_u, data2)
        
        # p(xt,at)
        pred = self.model(states, log=True)  # log parameter allows to use the log_softmax function of pytorch
        actions = actions.view(-1, 1, 1).expand(self.batch_size, 1, self.nb_atoms)
        q0 = pred.gather(1, actions).squeeze()
     
        # Cross-entropy loss
        loss = -(m * q0).sum(1).sum()
        
        return loss  
