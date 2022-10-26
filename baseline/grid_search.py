from array import ArrayType, array
from copy import copy
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
from regex import B, P
from sklearn.linear_model import LinearRegression
import os
import torch
from scipy.stats import uniform
from copy import deepcopy
import itertools
import sys
head, tail = os.path.split(__file__)
sys.path.insert(0, os.path.join(head, '..'))
from envs.linear_fitting import Linear_model


class grid_search():

    def __init__(self):
        pass

    def set_seed(self,seed):

        torch.manual_seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    def get_iter_comb(self,param_grid:dict):

        r = [] 
        list_para = list(param_grid.values())
        b= deepcopy(list_para)
        result = b[0]
        for i in range(len(list_para)):
            if len(b) != 1:
                b.pop(0)
                result = itertools.product(result,b[0])
            else:
                break

        return result

    def grid_search(self,param_grid:dict):
        # self.set_seed(24) 


        sample_locate = []
        hyperparameters =[]

        hyperparameters = self.get_iter_comb(param_grid)

        return hyperparameters



def loss(x,y,k,b):
        
    total_cost = 0
    M = len(x)

    for i in range(M):

        total_cost += (y[i] - k* x[i] - b)**2

    return total_cost/M


if __name__ == "__main__":

    MAX_EVALS = 1000000
        # 记录用
    best_score = -np.inf
    best_hyperparams = []

    param_grid = {
        'k': list(np.arange(0,10,1e-2)),
        'b': list(np.arange(0,10,1e-2))
        }

    model = Linear_model()
    x,y = model.data_generate()
    coef = []
    intercept = 0
    result =[]



    search_model = grid_search()

    hp = search_model.grid_search(param_grid=param_grid)

    step = 0

    for [k,b] in hp:
        step += 1
        if step > MAX_EVALS:
            break
        score = -loss(x,y,k,b)
        if score > best_score:
            best_hyperparams = [k,b]
            best_score = score
            # torch.save(model.state_dict(), "best_model.pt")

    y_pred,r,coef,intercept = model.model_fitting(x,y)
    print(coef,intercept)
    print(best_hyperparams)


