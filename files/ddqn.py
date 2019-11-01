# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 02:03:07 2019

@author: Clement_X240
"""

from dqn import DQN

# The only difference between the DQN and the DDQN is the loss function
# So the DDQN class inherit from the DQN class


class DDQN(DQN):
    
    def __init__(self, env, model, target_model, config, name_agent="ddqn"):
        super(DDQN, self).__init__(env, model, target_model, config, name_agent=name_agent)
        
    def loss(self):
        """ 
            the loss is equal to:
                    Rt+1+γt+1qθ_barre(St+1,argmax qθ(St+1,a′))−qθ(St,At))^2
        """
        # minibatch of experiences
        states, actions, rewards, next_states, finish = self.replay_buffer.sample()
        
        # transform the actions in integer
        actions = actions.long() 
        
        # gather function is used to take in the torch tensor the proba at the actions' indices
        
        # squeeze and unsqueeze is used to reshape the tensor at the good dimensions for the computation
        # probability next_states (using the current model) considering the action made
        # qθ(St,At)
        q0 = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1) 
        
        # take the most likely next-next state using the current model (considering we are in the next_state
        # position)
        # argmax qθ(S_t+1,a′)
        next_q0_model = self.model(next_states).argmax(1).unsqueeze(1) 
        
        # *(1-finish) if the episode is already finished, the action can't be taken, so it has to be equal to zero
        # qθ_barre(S_t+1, argmax qθ(S_t+1,a′))
        max_next_q0_target = self.target_model(next_states).gather(1, next_q0_model).squeeze(1) * (1-finish)
        
        Rt_gamma_max = (rewards + self.gamma * max_next_q0_target)
        
        loss = (q0 - Rt_gamma_max).pow(2).sum()
        
        return loss
