# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 02:13:29 2019

@author: Clement_X240
"""

from prioritized import Prioritized
import numpy as np
import torch


class Rainbow(Prioritized):
    
    def __init__(self, env, model, target_model, config):
        super(Rainbow, self).__init__(env, model=model, target_model=target_model, config=config, name_agent ="rainbow-dqn")

        # discrete support (for the aproximating distribution dt)
        self.Z = torch.linspace(config.Z_min, config.Z_max, config.nb_atoms)
        self.Z_min = config.Z_min
        self.Z_max = config.Z_max
        self.nb_atoms = config.nb_atoms

        self.nb_steps = config.nb_steps
        self.temp_nstep = []

    def action(self, state):
        """
            Select an action to take with an epsilon greedy strategy.
            
            Epsilon greedy strategy: with probability ϵ select a random action,
            otherwise select a = argmax Q(st,a)

            Params
            ------
            state: 
                current state

            Returns
            -------
            action: int
                The action the agent will take.
        """
        
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.nb_actions)
            
        else:
            prob_action = self.model(state) * self.Z
            action = prob_action.sum(2).argmax()
            return action.argmax().item()

    def save_temp(self, state, action, reward, next_state, finish):
        self.temp_nstep.append((state, action, reward, next_state, finish))
        
    def save_data(self, state, action, reward, next_state, finish):
        """
            Saving the experiene is a bit different as we are saving St+n not St+1.
            and the reward are:
                   R^(n)_t = somme_k^(n-1) (γ_k^t * (Rt+k+1))
            Params
            ------
            state: S_t
            action: A_t
            reward: R_t+1
            next_state: S_t+1
            finish: 1 if the action lead to the end of the episode, 0 otherwise
        """  
        
        if finish == 1:
            # if the episode, we need to save the remaining experience
            self.save_temp(state, action, reward, next_state, finish)
            while len(self.temp_nstep)>0:
                self.save_reward(reward, next_state, finish)
        else: 
            self.save_temp(state, action, reward, next_state, finish)
        
            # saving the experience if we have done the necessary number of step
            if len(self.temp_nstep)==self.nb_steps:
                self.save_reward(reward, next_state, finish)
        
    def save_reward(self, reward, next_state, finish):
            reward = 0
            
            #  compute somme_k^(n-1) (γ_k^t * (Rt+k+1))
            for i in range(len(self.temp_nstep)):
                reward += self.temp_nstep[i][2] * (self.gamma**i)
                
            # delete the experience of the time t from the list
            state, action, _, _, _ = self.temp_nstep.pop(0)
            
            self.replay_buffer.append_data(state, action, reward, next_state, finish)
            
    def loss(self):
        
        """
            the loss is equal to:
                DKL(Φzd't||dt)
                
                with d't≡(R_t(n)+γ_t(n) z,pθ(St+n,a∗_t+n))                        
        """
        
        states, actions, rewards, next_states, finish, indices, weight = self.replay_buffer.sample()
        actions = actions.long()
        
        # Q(xt+n,a) := somme_i(zi pi(xt+n,a))
        next_proba = self.target_model(next_states) * self.Z

        # a* = argmax_a(Q(xt+n,a))
        max_next_q0 = next_proba.sum(2).argmax(1).view(-1,1,1).expand(-1, -1, self.nb_atoms)
        
        # redimission the rewards, finish and the support to avoid the loop
        rewards = rewards.unsqueeze(1).expand(-1, self.nb_atoms)
        finish = finish.unsqueeze(1).expand(-1, self.nb_atoms)
        Z = self.Z.unsqueeze(0).expand(self.batch_size, -1)

        # compute the projection in the support of the support (by using clamp function)
        Tz = (rewards + (1-finish) * (self.gamma**self.nb_steps) * Z).clamp(min=self.Z_min, max=self.Z_max)
        b = (Tz - self.Z_min)/((self.Z_max - self.Z_min) / (self.nb_atoms - 1))
        l = b.floor().long()
        u = b.ceil().long()
        
        temp = torch.linspace(0, (self.batch_size - 1) * self.nb_atoms, self.batch_size).long().unsqueeze(1).expand(self.batch_size, self.nb_atoms)
        
        index_l = (temp + l).view(-1)
        index_u = (temp + u).view(-1)
        
        # p(xt+n,a∗)
        next_proba_action = next_proba.gather(1, max_next_q0).squeeze(1)
        
        data1 = (next_proba_action * (u.float() - b)).view(-1)
        data2 = (next_proba_action * (b - l.float())).view(-1)
        
        # the function index_add permit to do m=m + ... without using a loop
        m = torch.zeros(self.batch_size, self.nb_atoms)
        m.view(-1).index_add_(0, index_l, data1)  
        m.view(-1).index_add_(0, index_u, data2)
        
        # p(xt,at)
        pred = self.model(states, log=True)
        actions = actions.view(-1,1,1).expand(self.batch_size, 1, self.nb_atoms)
        q0 = pred.gather(1, actions).squeeze(1)
     
        # Cross-entropy loss
        loss = (-m * q0).sum(1) 
        
        # update the priority of the buffer
        self.replay_buffer.add_p(indices, loss.detach().numpy())
         
        loss = loss * weight
        return loss.mean()       

    def train(self, nb_episode):
        state = self.environment.reset()
        sum_reward_game = 0
        
        state = torch.from_numpy(state).float()
        
        i_episode = 0
        i = 0
        while i_episode<nb_episode:
            # print("i: ",i)
            action = self.action(state)
            next_state, reward, finish, info = self.environment.step(action)
            sum_reward_game += reward
            self.update_epsilon(i)

            if finish == 1:
                reward = 0
                i_episode += 1
                
            self.save_data(state, action, reward, next_state, finish)
            
            if finish == 1:
                state = self.environment.reset()
                state = torch.from_numpy(state).float()
                self.rewards.append(sum_reward_game)
                sum_reward_game = 0
            else:
                state = torch.from_numpy(next_state).float()
            
            if i > self.batch_size:
                self.do_one_step()
                
            if i % self.update_nb_iter == 0:
                self.update_model()
                
            i += 1