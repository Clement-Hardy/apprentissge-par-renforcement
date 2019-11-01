# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 02:15:53 2019

@author: Clement_X240
"""
import numpy as np
import torch


class ReplayBuffer(object):
    
    def __init__(self, max_data, batch_size):
        
        self.max_data = max_data      # memory size of the buffer
        self.data = [] 
        self.pos = 0                  # the current position in the list containing the data
        self.batch_size = batch_size  # the minibatch size

    def append_data(self, state, action, reward, next_state, finish):
        """
            Store a new experience in the buffer.

            Params
            ------
            state: S_t
            action: A_t
            reward: R_t+1
            next_state: S_t+1
            finish: 1 if the action lead to the end of the episode, 0 otherwise
        """        
        if len(self.data) < self.max_data:
            # if the list isn't already full we simply push the result 
            # in the end of the list
            self.data.append(tuple((state, action, reward,
                                    next_state, finish)))
        else:
            # otherwise we replace the data at the position "pos" of the list
            self.data[self.pos] = tuple((state, action, reward, next_state, finish))
        self.pos += 1
        self.pos %= self.max_data  # when we arrive at the end of the list we restart at the beginning of the list
        # like that the old result is deleted from the list.
        
    def sample(self):
        """
            Sample a minibatch from Replay Buffer.

            Returns
            -------
            states: float tensor
                Batch of observations
            actions: float tensor
                Batch of actions executed given states
            rewards: float tensor
                Rewards received as results of executing actions
            next_states: np.array
                Next set of observations seen after executing actions
            finish: float tensor
                finish[i] = 1 if executing actions[i] resulted in the end of an 
                episode and 0 otherwise.
            """
        
        # the experiences of the minibatch are choosed randomly (the minibatch has the size batch_size)
        indices = np.random.randint(0, len(self.data), self.batch_size)
        states, actions, rewards, next_states, finishs = [], [], [], [], []
        
        # we add the experience in the minibatch
        for i in indices:
            states.append(self.data[i][0])
            actions.append(self.data[i][1])
            rewards.append(self.data[i][2])
            next_states.append(self.data[i][3])
            finishs.append(self.data[i][4])
        
        # converting numpy arrays to float tensors (pytorch can't work with numpy array)
        return torch.stack(states), torch.FloatTensor(actions), torch.FloatTensor(rewards), \
            torch.FloatTensor(next_states),  torch.FloatTensor(finishs)

    
class PrioritizedReplayBuffer(object):
    
    def __init__(self, max_data, batch_size, w, beta_final, beta_start, beta_decay):
        
        self.max_data = max_data        # memory size of the buffer
        self.data = []
        self.pos = 0                    # the current position in the list containing the data
        self.batch_size = batch_size    # the minibatch size
        self.w = w                      # Prioritization exponent
        self.p = np.insert(np.zeros(self.max_data-1, dtype=float), 0, 1.0)
        self.call_update_beta = 0       # number of time the function to update beta has been call

        self.beta_final = beta_final
        self.beta_start = beta_start
        self.beta_decay = beta_decay
        
    def append_data(self, state, action, reward, next_state, finish):
        """
            Store a new experience in the buffer.

            Params
            ------
            state: S_t
            action: A_t
            reward: R_t+1
            next_state: S_t+1
            finish: 1 if the action lead to the end of the episode, 0 otherwise
        """          
        if len(self.data) < self.max_data:
            self.data.append(tuple((state, action, reward, next_state, finish)))
        else:
            self.data[self.pos] = tuple((state, action, reward, next_state, finish))
        
        self.p[self.pos] = np.max(self.p)
        self.pos += 1
        self.pos %= self.max_data
        
    def add_p(self, indices, p):
        """
            Update priority 
            
            Params
            ------
            indices: tuple
                Data of the experience: (S_t, A_t, R_t+1, S_t+1, end_of_game)
        """
        self.p[indices] = np.power((np.abs(p) + 1e-6), self.w)  # we add a small number so that every experience has
        # a non zero probability to be sampled

    def update_beta(self):
        """
            update the probability of exploration
            
            params:
                i:int
                    the current iteration of the agent
        """
        
        self.beta = self.beta_final + (self.beta_start - self. beta_final) * \
            np.exp(-1. * self.call_update_beta / self.beta_decay)
        self.call_update_beta += 1
        
    def sample(self):
        """
            Sample a minibatch from Replay Buffer.

            Returns
            -------
            states: float tensor
                Batch of observations
            actions: float tensor
                Batch of actions executed given states
            rewards: float tensor
                Rewards received as results of executing actions
            next_states: np.array
                Next set of observations seen after executing actions
            finish: float tensor
                finish[i] = 1 if executing actions[i] resulted in the end of an 
                episode and 0 otherwise.
            indices: numpy array
                Indices in buffer of sampled experiences.
            """
        
        proba = self.p/np.sum(self.p)  # probability of each experience to be taken
        
        # sample the experiences of the minibatch following the probability above
        indices = np.random.choice(a=np.arange(0, self.max_data), size=self.batch_size, replace=False, p=proba)
        states, actions, rewards, next_states, finishs = [], [], [], [], []
        
        self.update_beta()
        
        weight = np.power((len(self.data)*proba[:len(self.data)]), -self.beta)
        weight /= np.min(weight)
        # we add the experience in the minibatch
        for i in indices:
            states.append(self.data[i][0])
            actions.append(self.data[i][1])
            rewards.append(self.data[i][2])
            next_states.append(self.data[i][3])
            finishs.append(self.data[i][4])
        
        # converting the numpy array to float tensor (pytorch can't work with numpy array)
        return torch.stack(states), torch.FloatTensor(actions), torch.FloatTensor(rewards), \
            torch.FloatTensor(next_states),  torch.FloatTensor(finishs), indices, torch.FloatTensor(weight[indices])
