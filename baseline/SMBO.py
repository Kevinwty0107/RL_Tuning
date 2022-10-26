from importlib.resources import path
from sklearn import datasets
from sklearn.preprocessing import normalize, scale
from hyperopt import fmin,tpe,hp,partial,Trials

from array import ArrayType, array
from copy import copy
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
from regex import B, P, X
from sklearn.linear_model import LinearRegression
import os
import torch
from scipy.stats import uniform
from copy import deepcopy
import itertools
import os, sys
head, tail = os.path.split(__file__)
sys.path.insert(0, os.path.join(head, '..'))
from envs.linear_fitting import Linear_model


model = Linear_model()
x,y = model.data_generate()






def hyperopt_model_score(params):

    
        total_cost = 0
        M = len(x)

        # k = random.choice(params['k'])
        # b= random.choice(params['b'])

        for i in range(M):


            total_cost += (y[i] - params['k'] * x[i] - params['b'])**2

        return total_cost/M



def f_model(params):
        acc = hyperopt_model_score(params)
        return acc


if __name__ == "__main__":
    
    MAX_EVALS = 100000
        # 记录用
    best_score = -np.inf
    best_hyperparams = []



    param_grid = {
    'k': hp.choice('k', range(0, 10)),
    'b': hp.choice('b', range(0, 10)),
    
    }

    model = Linear_model()
    x,y = model.data_generate()
    coef = []
    intercept = 0
    result =[]


    y_pred,r,coef,intercept = model.model_fitting(x,y)
    M = len(x)
    total_cost = 0
    for i in range(M):
        total_cost += (y[i] -coef * x[i] - intercept)**2


    print(total_cost/M)
    print(coef,intercept)
    print(best_hyperparams)
    trials = Trials()
    best = fmin(f_model, param_grid, algo=tpe.suggest, max_evals=1000, trials=trials)
    print('best:')
    print(best)


