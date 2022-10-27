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
from baseline import SMBO
from hyperopt import fmin,tpe,hp,partial,Trials
from Index.PGM import Parameter_change
from baseline.random_search import random_search
from baseline.grid_search import grid_search
from tqdm import tqdm
from utils import utils
from agents import TD3
from envs.env import LinearFitting, PGMIndex
from envs.linear_fitting import Linear_model
from agents import DDPG
from agents import dqn


def eval_policy(policy, data, eval_episodes=4):
    eval_env = PGMIndex(data)
    eval_env.reset()
    state = []
    best_state = [] 
    best_reward = -np.inf
    reward_list = []
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        for _ in range(4):
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            reward_list.append(reward)
            if reward >= best_reward:
                best_reward = reward
                best_state = state
            
    avg_reward = avg_reward/(eval_episodes*4)

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.5f} parameter: {best_state} best reward:{best_reward:.5f}")
    print("---------------------------------------")
    return [avg_reward,best_state,best_reward]



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--RL_policy", default="TD3") # Policy name (TD3, DDPG, SAC or DDPG)
    # parser.add_argument("--data_file", default='data_0')
    parser.add_argument("--search_method", default='RL', help="method to use")
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=30, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=30, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=3e3, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", default=False)              # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--sample_mode",default="random_local")     # random_local: local + global sampling, random: only global sampling


    args = parser.parse_args()

    evaluations = []

    for i in range(30):

        data_name = "data_" + f"{i}"
        data_file_name = data_name+".txt"

        env_name = "PGMIndex"
        print("---------------------------------------")
        print(f"Policy: {args.RL_policy}, Env: {env_name}, Seed: {args.seed}")
        print("---------------------------------------")


        if not os.path.exists(f"./results/{args.search_method}"):
            os.makedirs(f"./results/{args.search_method}")


        if args.save_model and not os.path.exists("./rlmodels"):
            os.makedirs("./rlmodels")



        env = PGMIndex(data_file_name)

        env.reset()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)


        state_dim = 2 # type: ignore
        action_dim = env.action_space.n  # type: ignore
        max_action =  env.action_space.n   # type: ignore

        kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        } 

            # Initialize policy
        if args.RL_policy == "TD3":
            # Target policy smoothing is scaled wrt the action scale
            kwargs["policy_noise"] = args.policy_noise * max_action
            kwargs["noise_clip"] = args.noise_clip * max_action
            kwargs["policy_freq"] = args.policy_freq
            policy = TD3.TD3(**kwargs)

        elif args.RL_policy == "DQN":

            kwargs_dqn = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            } 

            policy = dqn.DQN(**kwargs_dqn)


        else:
            policy = DDPG.DDPG(**kwargs)

    
        file_name = f"{args.RL_policy}_PGMIndex_{args.seed}"

        if args.load_model != "":
            policy_file = file_name if args.load_model == "default" else args.load_model
            policy.load(f"./rlmodels/{policy_file}")

    

        # Evaluate untrained policy

        evaluations.append(eval_policy(policy, data_file_name))
        np.save(f"./results/{args.search_method}/{file_name}_{data_name}", evaluations)
        if args.save_model: policy.save(f"./rlmodels/{file_name}")


    