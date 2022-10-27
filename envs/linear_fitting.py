from array import ArrayType, array
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
from sklearn.linear_model import LinearRegression
import os
import torch
from scipy.stats import uniform


class Linear_model():

    def __init__(self):
        super().__init__()


    def data_generate(self):
        """ Generate original data tuples"""

        xdata = np.ones((20))
        for i in range(20):
            xdata[i] = np.random.random()

        ydata = np.zeros((20))
        for k in range(20):
            random1 = np.random.random()
            s = random.randint(1,3)
            ydata[k] = s * xdata[k]+random1

        return xdata, ydata



    def data_learning_segmentation(self,xdata,ydata):

        """Generate Training and testing tuples """ 

        x_train = xdata[0:70]
        x_test = xdata[70:-1]
        y_train = ydata[0:70]
        y_test = ydata[70:-1]
    

        return x_train, x_test,y_train,y_test


    def model_fitting(self,x,y):

        """Linear fitting, output intercept and coef"""
        
        model = LinearRegression() # 构建线性模型
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        model.fit(x, y) # 自变量在前，因变量在后
        predicts = model.predict(x) # 预测值
        R2 = model.score(x, y) # 拟合程度 R2
        # print('R2 = %.2f' % R2) # 输出 R2
        coef = model.coef_ # 斜率
        intercept = model.intercept_ # 截距

        return predicts, R2, coef,intercept



# 这里不固定 random 模块的随机种子，因为 random 模块后续要用于超参组合随机组合。
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
    	torch.cuda.manual_seed_all(seed)
    	# torch.backends.cudnn.deterministic = True
    	# torch.backends.cudnn.benchmark = False



def loss(x,y,k,b):

    total_cost = 0
    M = len(x)

    for i in range(M):

        total_cost += (y[i] - k* x[i] - b)**2

    return total_cost/M


def gauss_kernal(sample_scale,sigma):
    tempa = sample_scale.reshape(-1, 1) 
    tempb = np.transpose(tempa)
    temp = tempa - tempb
    gauss = math.exp(- temp**2 / (2 * sigma**2))
    return gauss


def random_search(x_train,y_train,discrete = False):
    set_seed(24) 
    param_grid = {
    'k': list(np.arange(0,10,1e-3)),
    'b': list(np.arange(0,10,1e-3))
    }
    MAX_EVALS = 10000

    # 记录用
    best_score = -np.inf
    best_hyperparams = []

    for i in range(MAX_EVALS):
        random.seed(i)
        if discrete == True:
            hyperparameters =  {k: random.sample(v,1)[0] for k,v in param_grid.items()}
            k = hyperparameters['k']
            b = hyperparameters['b']
        else:
            sigma_k = 0.2
            sigma_b = 0.2
            sample_locate_k = np.linspace(param_grid['k'][0],param_grid['k'][-1], 1)
            sample_locate_b = np.linspace(param_grid['b'][0],param_grid['b'][-1], 1)
            # sample_locate = np.linspace(sample_scale[0],sample_scale[1],number_dot)
            # sample_locate = np.linspace(sample_scale[0],sample_scale[1],number_dot)
            k = np.random.normal(np.zeros(1),gauss_kernal(sample_locate_k, sigma_k),1) 
            b = np.random.normal(np.zeros(1),gauss_kernal(sample_locate_b, sigma_b),1) 

        score = -loss(x_train,y_train,k,b)
        if score > best_score:
            best_hyperparams = [k,b]
            best_score = score
            # torch.save(model.state_dict(), "best_model.pt")
    
    return best_hyperparams


from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression

 
class test():

    def __init__(self,k=1,b=1):

        self.k = k
        self.b = b
        
    def fit(self,x,y):
        model = LogisticRegression(intercept_scaling=self.k,verbose=self.b) # 构建线性模型
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        
        s = model.fit(x, y) # 自变量在前，因变量在后

        return s






        
 
def random_search_sklearn(x_train,y_train):   
 
    clf1 = test()
 
    param_dist = {
        'k': uniform(0,10),
        'b':uniform(0,10),
        }
 
    custom_score = make_scorer(loss, greater_is_better=False)

    rdm = RandomizedSearchCV(clf1,param_dist,cv = 3,scoring = custom_score,n_iter=300,n_jobs = -1)
 
    rdm.fit(x_train,y_train)

    best_estimator = rdm.best_estimator_

    return best_estimator




import time

if __name__ == "__main__":



    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="both")        

    model = Linear_model()
    x,y = model.data_generate()
    coef = []
    intercept = 0
    result =[]

    args = parser.parse_args()
    if args.method == "fitting":

        # print(np.shape(x),np.shape(y))
        # plot predicted y
        y_pred,r,coef,intercept = model.model_fitting(x,y)
        print(coef,intercept)


    if args.method == "random":
        print(random_search(x,y))

    if args.method == "both":

        # print(np.shape(x),np.shape(y))
        # plot predicted y
        start = time.monotonic()
        y_pred,r,coef,intercept = model.model_fitting(x,y)
        end = time.monotonic()

        time_used = end - start
        print(coef,intercept, 'time = %f'%time_used)

        start = time.monotonic()
        result = random_search(x,y,discrete = False)
        end = time.monotonic()

        time_used = end - start

        print(result, 'time = %f'%time_used )
        




    #plot fitting curve
    x_axis = np.linspace(0,1, num=50)
    y_linear = [ j *coef[0]+intercept for j in x_axis ]

    plt.figure(0)
    plt.scatter(x,y)
    plt.plot(x_axis,y_linear,label="fitting") 
    intercept = np.ndarray.flatten(np.array(result[0]))
    coef = np.ndarray.flatten(np.array(result[1]))
    y_linear = [ j *intercept[0]+coef[0]  for j in x_axis ]
    plt.plot(x_axis,y_linear,label="random_search")
    plt.legend()
    plt.show()

