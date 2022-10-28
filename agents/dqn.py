import copy
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from gym import wrappers
from gym import spaces
from abc import ABCMeta, abstractmethod


from collections import namedtuple
import random

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device=torch.device("cpu")

# torch.FloatTensor=torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor #如果有GPU和cuda，数据将转移到GPU执行
# torch.LongTensor=torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

class Net(nn.Module):
    def __init__(self, state_dim,action_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50,action_dim)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class DQN:
    def __init__(self,
    state_dim,
    action_dim,
    BATCH_SIZE = 32,            
    LR = 0.01,                  
    EPSILON = 0.9,               
    GAMMA = 0.9,                
    TARGET_REPLACE_ITER = 100, 
    MEMORY_CAPACITY= 15 ):
        self.net,self.target_net=Net(state_dim,action_dim).to(device),Net(state_dim,action_dim).to(device)
        
        self.learn_step_counter=0
        self.state_dim = state_dim
        self.action_dim= action_dim
        self.BATCH_SIZE = BATCH_SIZE
        self.EPSILON =EPSILON
        self.GAMMA = GAMMA
        self.TARGET_REPLACE_ITER = TARGET_REPLACE_ITER
        self.memory_counter=0
        self.MEMORY_CAPACITY= MEMORY_CAPACITY
        self.memory = np.zeros((MEMORY_CAPACITY, state_dim * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        

    def select_action(self, x):
        x = torch.FloatTensor(x.reshape(1, -1)).to(device)
        # input only one sample
        if np.random.uniform() < self.EPSILON:  
            actions_value = self.net.forward(x)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
        else:   
            action = np.random.randint(0, self.action_dim)
            
        return action
        
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def train(self):
        # target parameter update
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_s = torch.FloatTensor(batch_memory[:, :self.state_dim]).to(device)
        batch_a = torch.LongTensor(batch_memory[:,self.state_dim:self.state_dim+1].astype(int)).to(device)
        batch_r = torch.FloatTensor(batch_memory[:, self.state_dim+1:self.state_dim+2]).to(device)
        batch_s_ = torch.FloatTensor(batch_memory[:, -self.state_dim:]).to(device)

        
        q = self.net(batch_s).gather(1, batch_a)  # shape (batch, 1)
        q_target = self.target_net(batch_s_).detach()     # detach from graph, don't backpropagate
        y = batch_r + self.GAMMA * q_target.max(1)[0].view(self.BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filename):
        torch.save(self.net.state_dict(), filename + "_q_net")
        torch.save(self.target_net.state_dict(), filename + "_target_net")
        torch.save(self.optimizer.state_dict(), filename + "_optimizer")

    def load(self, filename):
        self.net.load_state_dict(torch.load(filename + "_q_net"))
        self.target_net.load_state_dict(torch.load(filename + "_target_net"))
        self.optimizer.load_state_dict(torch.load(filename + "_optimizer"))

		