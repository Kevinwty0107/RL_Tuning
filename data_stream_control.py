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
import os 


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_method", default='grid_search', help="method to use")
    parser.add_argument("--Index", default= "PGM")                  # Policy name (TD3, DDPG, SAC or OurDDPG)
    parser.add_argument("--data_file", default='data_0')# Time steps initial random policy is used


    args = parser.parse_args()

    data_file_name = args.data_file + ".txt"


    if args.Index == "PGM":

            Parameter_change.updateFile("./Index/PGM/index_test.cpp",252,1)
            os.system('g++ ./Index/PGM/index_test.cpp  -w -std=c++17 -o ./Index/PGM/exe_pgm_index')
            os.system(f'./Index/PGM/exe_pgm_index ./data/{data_file_name}')

            f = open("runtime_result.txt",encoding="utf-8")
            cost = float(f.read())
            f.close

            file_name = "result"+ f"_{args.data_file}"

            result = []
            result.append(cost)
            print(cost)

            np.save(f"./results/{args.search_method}/{file_name}", result)