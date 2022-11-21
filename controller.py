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
from hyperopt import fmin,tpe,hp,partial,Trials,space_eval
from Index.PGM import Parameter_change
from baseline.random_search import random_search
from baseline.grid_search import grid_search
from tqdm import tqdm
from utils import utils
from agents import TD3
from envs.env import LinearFitting, PGMIndex
from envs.linear_fitting import Linear_model
from agents import DDPG



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_method", default='random_search', help="method to use")
    parser.add_argument("--Index", default= "PGM")                  # Policy name (TD3, DDPG, SAC or OurDDPG)
    parser.add_argument("--dynamic", default=False, type=bool)              # dynamic data or static data
    parser.add_argument("--data_file", default='data_0')# Time steps initial random policy is used


    args = parser.parse_args()

    data_file_name = args.data_file + ".txt"


    if args.search_method == "default":

        if args.Index == "PGM":

            Parameter_change.updateFile("./Index/PGM/index_test.cpp",64,4)
            os.system('g++ ./Index/PGM/index_test.cpp  -w -std=c++17 -o ./Index/PGM/exe_pgm_index')
            os.system(f'./Index/PGM/exe_pgm_index ./data/{data_file_name}')

            f = open("runtime_result.txt",encoding="utf-8")
            cost = float(f.read())
            f.close

            if not os.path.exists(f"./results/{args.search_method}"):
                os.makedirs(f"./results/{args.search_method}")

            file_name = "result"+ f"_{args.data_file}"

            result = []
            result.append(cost)
            print(cost)

            np.save(f"./results/{args.search_method}/{file_name}", result)


    if args.search_method == "BO":

        def hyperopt_model_score(params):

            if args.Index == "PGM":

                Parameter_change.updateFile("./Index/PGM/index_test.cpp",params['epsilon'],params['ER'])
                os.system('g++ ./Index/PGM/index_test.cpp  -w -std=c++17 -o ./Index/PGM/exe_pgm_index')
                os.system(f'./Index/PGM/exe_pgm_index ./data/{data_file_name}')

            # other Index in progress

            f = open("runtime_result.txt",encoding="utf-8")
            cost = float(f.read())
            f.close
            
            return cost 

        def f_model(params):
            acc = hyperopt_model_score(params)
            return acc

        param_grid = {
            'epsilon': hp.choice('epsilon', range(1,8000) ),
            'ER': hp.choice('ER', range(1,20))}

        trials = Trials()
        start = time.monotonic()
        best = fmin(f_model, param_grid, algo=tpe.suggest, max_evals=1000, trials=trials)
        end = time.monotonic()
        best_hyp = space_eval(param_grid, best)
        time_tuning = end- start
        print('best:')
        print(best_hyp)
        print('time used:')
        print(time_tuning)

        if not os.path.exists(f"./results/{args.search_method}"):
            os.makedirs(f"./results/{args.search_method}")

        file_name = "result"+ f"_{args.data_file}"

        result = []
        result.append(hyperopt_model_score(best_hyp))
        result.append(time_tuning)
        result.append(best_hyp)

        np.save(f"./results/{args.search_method}/{file_name}", result)


    if args.search_method == "random_search":

        MAX_EVALS = 1000
        # 记录用
        best_score = np.inf
        best_hyperparams = []

        param_grid = {
            'epsilon': list(np.arange(1,8000)),
            'b': list(np.arange(1,20))
            }

        result =[]

        def model_loss(epsilon,er):
    
            if args.Index == "PGM":

                Parameter_change.updateFile("./Index/PGM/index_test.cpp",epsilon,er)
                os.system('g++ ./Index/PGM/index_test.cpp  -w -std=c++17 -o ./Index/PGM/exe_pgm_index')
                os.system(f'./Index/PGM/exe_pgm_index ./data/{data_file_name}')

            # other Index in progress

            f = open("runtime_result.txt",encoding="utf-8")
            cost = float(f.read())
            f.close
            
            return cost 

        search_model = random_search()
        start = time.monotonic()

        for i in tqdm(range(MAX_EVALS)):

    
            [epsilon,er] = search_model.random_search(param_grid=param_grid,discrete = True)


            score = model_loss(epsilon,er)

            print(f"---random searching on {i}th iteration, run time is {score} ms")
            if score < best_score:
                best_hyperparams = [epsilon,er]
                best_score = score
            # torch.save(model.state_dict(), "best_model.pt")

        end = time.monotonic()
        time_tuning = end- start
        print('best:')
        print(best_hyperparams)
        print('time used:')
        print(time_tuning)

        if not os.path.exists(f"./results/{args.search_method}"):
            os.makedirs(f"./results/{args.search_method}")

        file_name = "result"+ f"_{args.data_file}"

        result = []
        result.append(model_loss(best_hyperparams[0],best_hyperparams[1]))
        result.append(time_tuning)
        result.append(best_hyperparams)

        np.save(f"./results/{args.search_method}/{file_name}", result)

    if args.search_method == "grid_search":
    
        MAX_EVALS = 1000
        # 记录用
        best_score = np.inf
        best_hyperparams = []

        param_grid = {
            'epsilon': list(np.arange(1,8000)),
            'er': list(np.arange(1,20))
            }

        result =[]

        def model_loss(epsilon,er):
    
            if args.Index == "PGM":

                Parameter_change.updateFile("./Index/PGM/index_test.cpp",epsilon,er)
                os.system('g++ ./Index/PGM/index_test.cpp  -w -std=c++17 -o ./Index/PGM/exe_pgm_index')
                os.system(f'./Index/PGM/exe_pgm_index ./data/{data_file_name}')

            # other Index in progress

            f = open("runtime_result.txt",encoding="utf-8")
            cost = float(f.read())
            f.close
            
            return cost 

        search_model = grid_search()

        step = 0
        hp_set = search_model.grid_search(param_grid=param_grid)
        start = time.monotonic()

        for [epsilon,er] in tqdm(hp_set):

            step += 1

            if step > MAX_EVALS:
                break
            score = model_loss(epsilon,er)

            print(f"---grid searching on {step}th iteration, run time is {score} ms")
            if score < best_score:
                best_hyperparams = [epsilon,er]
                best_score = score
            # torch.save(model.state_dict(), "best_model.pt")

        end = time.monotonic()
        time_tuning = end- start
        print('best:')
        print(best_hyperparams)
        print('time used:')
        print(time_tuning)

        if not os.path.exists(f"./results/{args.search_method}"):
            os.makedirs(f"./results/{args.search_method}")

        file_name = "result"+ f"_{args.data_file}"

        result = []
        result.append(model_loss(best_hyperparams[0],best_hyperparams[1]))
        result.append(time_tuning)
        result.append(best_hyperparams)

        np.save(f"./results/{args.search_method}/{file_name}", result)






        



        


        

        
        


