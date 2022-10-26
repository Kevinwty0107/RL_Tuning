from array import ArrayType, array
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
from regex import B
from sklearn.linear_model import LinearRegression
import os
import torch
from scipy.stats import uniform
import sys
head, tail = os.path.split(__file__)
sys.path.insert(0, os.path.join(head, '..'))
from envs.linear_fitting import Linear_model

class random_search():

    def __init__(self):
        pass

    def set_seed(self,seed):

        torch.manual_seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    def gauss_kernal(self,sample_scale,sigma):
        tempa = sample_scale.reshape(-1, 1) 
        tempb = np.transpose(tempa)
        temp = tempa - tempb
        gauss = math.exp(- temp**2 / (2 * sigma**2))
        return gauss

    def random_search(self,param_grid:dict,discrete = False):
        # self.set_seed(24) 


        sample_locate = []
        hyperparameters ={}

        if discrete == True:
            hyperparameters =  {k: random.sample(v,1)[0] for k,v in param_grid.items()}

        else:
            sigma = 0.2
            for key in param_grid.keys():
                sample_locate.append(np.linspace(param_grid[key][0],param_grid[key][-1], 1))
                v = np.random.normal(np.zeros(1),self.gauss_kernal(sample_locate[-1], sigma),1) 
                hyperparameters[key] = v
    
        return list(hyperparameters.values())





if __name__ == "__main__":

    MAX_EVALS = 100000
        # 记录用
    best_score = -np.inf
    best_hyperparams = []

    param_grid = {
        'k': list(np.arange(0,10,1e-3)),
        'b': list(np.arange(0,10,1e-3))
        }

    model = Linear_model()
    x,y = model.data_generate()
    coef = []
    intercept = 0
    result =[]

    def loss(x,y,k,b):
        
        total_cost = 0
        M = len(x)

        for i in range(M):

            total_cost += (y[i] - k* x[i] - b)**2

        return total_cost/M

    for i in range(MAX_EVALS):

        search_model = random_search()


        [k,b] = search_model.random_search(param_grid=param_grid,discrete = False)


        score = -loss(x,y,k,b)
        if score > best_score:
            best_hyperparams = [k,b]
            best_score = score
            # torch.save(model.state_dict(), "best_model.pt")

    y_pred,r,coef,intercept = model.model_fitting(x,y)
    print(coef,intercept)
    print(best_hyperparams)


