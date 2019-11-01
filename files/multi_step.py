# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 02:08:04 2019

@author: Clement_X240
"""
from dqn import DQN


class MultiStep(DQN):
    
    def __init__(self, env, model, target_model, config, name_agent="multistep-dqn"):
        super(MultiStep, self).__init__(env, model, target_model, config, name_agent=name_agent)
        
        self.nb_steps = config.nb_steps
        self.temp_nstep = []
        
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
        
        self.save_temp(state, action, reward, next_state, finish)
        
        if finish == 1:
            # if the episode, we need to save the remaining experience
            while len(self.temp_nstep)>0:
                self.save_reward(reward, next_state, finish)
        
        # saving the experience if we have done the necessary number of step
        if len(self.temp_nstep) == self.nb_steps:
            self.save_reward(reward, next_state, finish)
        
    def save_reward(self, reward, next_state, finish):
            reward = 0
            #  compute somme_k^(n-1) (γ_k^t * (Rt+k+1))
            for i in range(len(self.temp_nstep)):
                reward += self.temp_nstep[i][2] * (self.gamma**i)
                
            # delete the experience of the time t from the list
            state, action, _, _, _ = self.temp_nstep.pop(0)
            
            # save the experience in the buffer with right reward and S_t+n
            self.replay_buffer.append_data(state, action, reward, next_state, finish)
            
    def loss(self):
        """ 
            the loss is equal to:
                    R^(n)_t+γ^(n)_t max_(a') qθ_barre(St+n,a')−qθ(St,At))2
        """        
        states, actions, rewards, next_states, finish = self.replay_buffer.sample()
        actions = actions.long()
        # qθ(St,At)
        q0 = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # max_(a') qθ_barre(St+n,a')
        max_next_q0 = self.target_model(next_states).max(1)[0] * (1-finish)
        
        Rt_gamma_max = (rewards + (self.gamma**self.nb_steps) * max_next_q0)
        
        loss = (q0 - Rt_gamma_max).pow(2).sum()
        
        return loss
