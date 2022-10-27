import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import optim
import torchvision.utils
import numpy as np
import random
import os
import gym
from agents import TD3



def eval_policy(policy, env_name, seed, eval_episodes=5):
    eval_env = gym.make(env_name)
    eval_env.reset()

    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done,  _ = eval_env.step(action)
            avg_reward += reward
            print(type(done))

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

if __name__=="__main__":
    
    env_name = "BipedalWalker-v3"
    env = gym.make(env_name)
    env.reset()
    
    state_dim = env.observation_space.shape[0]  # type: ignore
    action_dim = env.action_space.shape[0]   # type: ignore
    max_action = float(env.action_space.high[0])  # type: ignore

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": 0.05,
        "tau": 0.01,
    }

    max_action = float(env.action_space.high[0])  # type: ignore


    # Initialize policy
    
        # Target policy smoothing is scaled wrt the action scale
    kwargs["policy_noise"] = 1* max_action
    kwargs["noise_clip"] = 2* max_action
    kwargs["policy_freq"] = 3
    policy = TD3.TD3(**kwargs)

    print(eval_policy(policy,env_name,seed=1))

