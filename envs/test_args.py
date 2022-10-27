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


def sum(a,b):
    return a+b

def main(args):
    print(sum(args.hidden_size,args.batch_size))

def sum(k,b):
    return k+b


def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.reset()

    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--hidden_size",default=800, type=int)
    parser.add_argument("--gpu_idx",default=None, type=int)
    parser.add_argument("--batch_size",default=256, type=int)
    parser.add_argument("--group",default=0, type=int)
    parser.add_argument("--epochs",default=20, type=int)

    args = parser.parse_args()

    print(sum(args.hidden_size,args.batch_size))

    k = 1
    b = 2

    kwargs = {"k":k,"b":b}
    print(sum(**kwargs))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(torch.device)
    print(device.type)

    a = []
    print(len(a))

