import gym
from gym import spaces
import numpy as np
import os

from torch import dtype
from Index.PGM import Parameter_change
from copy import deepcopy
import itertools

class LinearFitting(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['console']}

    def __init__(self, train_data):
        super(LinearFitting, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.x = train_data[0]
        self.y = train_data[1]
        
        self.para_dict = {'k': 1, 'b': 2} # slope and intercept
        self.action_space = spaces.Box(low=np.array([0,0],dtype=np.float32),
                               high=np.array([5,5],dtype=np.float32),
                               )
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=5,
                                            shape=(2,), dtype=np.float32)

    def step(self, action):

        self.para_dict["k"] = action[0]
        self.para_dict["b"] = action[1]
        reward = -self.loss()
        done = False

        info = {}

        return np.array([self.para_dict["k"],self.para_dict["b"]]).astype(np.float32) , reward, done, info

    def loss(self):

        total_cost = 0
        M = len(self.x)

        for i in range(M):

            x = self.x[i]
            y = self.y[i]

            total_cost += (y - self.para_dict["k"]* x - self.para_dict["b"]) ** 2

        return total_cost/M

    def reset(self):

        self.para_dict["k"] = 1
        self.para_dict["b"] = 2

        return np.array([self.para_dict["k"],self.para_dict["b"]]).astype(np.float32)  # reward, done, info can't be included

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

        pass

    def close (self):
        pass

class PGMIndex(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['console']}

    def __init__(self, data_file_name):
        super(PGMIndex, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:

        self.data_file_name = data_file_name
        
        self.para_dict = {'epsilon': 64, 'er': 4} # PGM tunable parameters

        self.action_dict = {
            'epsilon': list(np.arange(1,4000,100)),
            'er': list(np.arange(0,20,1))
            }

        action_comb =  itertools.product(self.action_dict["epsilon"],self.action_dict["er"])
        self.list_action_comb = [k for k in action_comb]

        action_dim = len(self.list_action_comb)
        state_dim = len(self.list_action_comb)

        self.action_space = spaces.Discrete(action_dim)

        # self.action_space = spaces.Tuple([spaces.Discrete(8000),spaces.Discrete(80)])

        # Example for using image as input (channel-first; channel-last also works):

        # space= {"epsilon": spaces.Box(0,8000,shape=(1,),dtype=int),"er":spaces.Box(5,100,shape=(1,),dtype=int)}

        self.observation_space = spaces.Discrete(state_dim)

        # self.observation_space = spaces.Box(low=1, high=80, shape=(2,), dtype=np.int32)

  

    def step(self, action):

        self.para_dict["epsilon"] = self.list_action_comb[action][0]
        self.para_dict["er"] =  self.list_action_comb[action][1]
        reward = -self.model_loss()
        done = False

        info = {}

        return np.array((self.para_dict["epsilon"],self.para_dict["er"])), reward, done, info

    def model_loss(self):
        
        Parameter_change.updateFile("./Index/PGM/index_test.cpp",self.para_dict["epsilon"],self.para_dict["er"])
        os.system('g++ ./Index/PGM/index_test.cpp  -w -std=c++17 -o ./Index/PGM/exe_pgm_index')
        os.system(f'./Index/PGM/exe_pgm_index ./Index/PGM/{self.data_file_name}')

            # other Index in progress

        f = open("runtime_result.txt",encoding="utf-8")
        cost = float(f.read())
        f.close
            
        return cost 

    def reset(self):

        self.para_dict["epsilon"] = 64
        self.para_dict["er"] = 4

        return np.array((self.para_dict["epsilon"],self.para_dict["er"])).astype(np.int32)  # reward, done, info can't be included

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

        pass

    def close (self):
        pass
    

