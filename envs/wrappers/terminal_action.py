import gym
from gym import spaces


class TerminalAction(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.action_space,
                          gym.spaces.Discrete), "only discrete action space can be wrapped in TerminalAction"
        self.action_space = spaces.Discrete(env.action_space.n + 1)
        self._last_info = None
        self._last_ob = None

    def step(self, action: int):
        # Action == 0 is no-op
        if action == 0:
            print('Action 0 - Need to reset')
            return self._last_ob, 0, True, self._last_info
        else:
            ob, reward, done, info = self.env.step(action - 1)
            self._last_info = info
            self._last_ob = ob
            return ob, reward, done, info

    def reset(self, **kwargs):
        return_info = kwargs.get('return_info')
        kwargs['return_info'] = True
        self._last_ob, self._last_info = self.env.reset(**kwargs)
        if return_info:
            return self._last_ob, self._last_info
        else:
            return self._last_ob


class DummyTerminalAction(gym.Wrapper):
    def __init__(self, env: gym.Env, copy_reward: bool = False):
        super().__init__(env)
        assert isinstance(env.action_space,
                          gym.spaces.Discrete), "only discrete action space can be wrapped in TerminalAction"
        self.action_space = spaces.Discrete(env.action_space.n + 1)
        self._last_info = None
        self._last_ob = None
        self._last_reward = 0
        self._copy_reward = copy_reward

    def step(self, action: int):
        # Action == 0 is no-op
        if action == 0:
            print('Action 0 - No op')
            if self._copy_reward:
                return self._last_ob, self._last_reward, False, self._last_info
            else:
                return self._last_ob, 0, False, self._last_info
        else:
            ob, reward, done, info = self.env.step(action - 1)
            self._last_info = info
            self._last_ob = ob
            self._last_reward = reward
            return ob, reward, done, info

    def reset(self, **kwargs):
        return_info = kwargs.get('return_info')
        kwargs['return_info'] = True
        self._last_ob, self._last_info = self.env.reset(**kwargs)
        if return_info:
            return self._last_ob, self._last_info
        else:
            return self._last_ob
