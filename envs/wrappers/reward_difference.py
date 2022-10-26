from typing import Callable, Union, Optional, Any

import gym
from gym import spaces


class RewardDifference(gym.Wrapper):
    def __init__(self, env: gym.Env, init_reward=Optional[Union[float, Callable[[Any, Any], float]]]):
        super().__init__(env)
        self._last_reward = None
        self._init_reward = init_reward

    def step(self, action: int):
        ob, reward, done, info = self.env.step(action)
        diff = reward - self._last_reward
        self._last_reward = reward
        return ob, diff, done, info

    def reset(self, **kwargs):
        return_info = kwargs.get('return_info')
        kwargs['return_info'] = True
        ob, info = self.env.reset(**kwargs)
        if isinstance(self._init_reward, float):
            self._last_reward = self._init_reward
        elif isinstance(self._init_reward, Callable):
            self._last_reward = self._init_reward(ob, info)
        else:
            self._last_reward = 0
        if return_info:
            return ob, info
        else:
            return ob
