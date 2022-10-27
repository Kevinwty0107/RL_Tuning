import pickle
from typing import Callable, Union, Optional, Any

import gym
from gym import spaces


class RecordRollouts(gym.Wrapper):
    def __init__(self, env: gym.Env, file_loc: str):
        super().__init__(env)
        self.file_loc = file_loc

        self.obs = []
        self.next_obs = []
        self.actions = []
        self.rewards = []
        self.ob = None

    def step(self, action: int):
        ob, reward, done, info = self.env.step(action)
        self.obs.append(self.ob)
        self.next_obs.append(ob)
        self.actions.append(action)
        self.rewards.append(reward)
        self.ob = ob
        if done:
            with open(self.file_loc, 'ab+') as f:
                pickle.dump((self.obs, self.next_obs, self.actions, self.rewards), f)
        return ob, reward, done, info

    def reset(self, **kwargs):
        return_info = kwargs.get('return_info')
        kwargs['return_info'] = True
        ob, info = self.env.reset(**kwargs)

        self.obs = []
        self.next_obs = []
        self.actions = []
        self.rewards = []
        self.ob = ob

        if return_info:
            return ob, info
        else:
            return ob
