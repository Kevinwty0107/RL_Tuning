import gym
import numpy as np
from gym import spaces
from gym.wrappers import TimeLimit


class TimeLimitWithOb(TimeLimit):
    def __init__(self, env: gym.Env, max_episode_steps=None):
        super().__init__(env, max_episode_steps)
        assert env.observation_space.shape is not None, \
            "TimeLimitWithOb can only wrap envs with observations with non-empty shapes"
        self.observation_space = spaces.Box(np.append(env.observation_space.low, np.float32(0.)),
                                            np.append(env.observation_space.high, np.float32(max_episode_steps)),
                                            (np.prod(env.observation_space.shape) + 1,))

    def step(self, action: int):
        ob, reward, done, info = super().step(action)
        ob = np.array(ob).flatten()
        ob = np.append(ob, self._max_episode_steps - self._elapsed_steps)
        return ob, reward, done, info

    def reset(self, **kwargs):
        ret = super().reset(**kwargs)
        if isinstance(ret, tuple):
            ob = np.append(np.array(ret[0]).flatten(), self._max_episode_steps)
            return (ob,) + ret[1:]
        else:
            ret = np.append(np.array(ret).flatten(), self._max_episode_steps)
            return ret
