import numpy as np
import sys
import os
sys.path.append(os.getcwd() + f"{os.sep}..")
from decision_tree.decision import DecisionTreeRegressor

class Adaboost:
    """
    Simplified version of adaboost regressor.
    
    --------------------
    Arguments:
        n_step: int
            Number of trees 
            
        learning_rate: float
            Coefficient of weight update
            
        max_depth: int
            The deepest level it can go.
    """
    def __init__(self, n_step:int=100, lr:float=1.0, max_depth=3):
        self.w = None
        self.beta = 1
        self.n = None
        self.trees = []
        self.betas = []
        self.p = None
        self.learning_rate = lr
        self.n_step = n_step
        self.max_depth = max_depth
    
    def get_loss(self, y_pred, y, mode=""):
        if mode == "l1":
            loss = np.abs(y_pred - y)
            
        elif mode == "l2":
            loss = (y_pred - y)**2

        d = np.max(loss)    
        L_i = loss/d
        return L_i
    
    def train(self, X, Y, mode="l1"):
        self.n = len(X)
        self.w = np.array([1/self.n]*self.n)
        self.p = self.w
      
        for t in range(self.n_step):
            DTR = DecisionTreeRegressor()
            index = np.random.choice(np.arange(self.n), self.n, self.p.tolist())
            DTR.train(X[index], Y[index], self.max_depth)
            Y_pred = DTR.predict(X[index])
            L_i = self.get_loss(Y_pred, Y[index], mode)
            La = self.p@L_i
            self.update(La, L_i)
            self.betas.append(self.beta)
            if self.beta > 1/2:
                continue
            self.trees.append(DTR)
        
    def update(self, La, L_i):
        """
        Update the strategy by loss on ith point.
        """
        self.beta = La/(1-La)
        for i in range(len(self.w)):
            self.w[i] *=self.learning_rate*self.beta**(1-L_i[i])
        self.p = self.w/sum(self.w)

    def predict(self, X):
        """
        median prediction
        
        return: np.array
            prediction of model
        """
        h_T = []
        T = len(self.trees)
        for t in range(T):
            h_t = self.trees[t].predict(X)
            h_T.append(h_t)

        h_T = np.c_[h_T]
        n = len(X)

        thres = np.sum([np.log(1/beta) for beta in self.betas])*.5
        # weighted median
        y_preds = []
        for i in range(n):
            sort_idx = np.argsort(h_T[:, i])
            beta_0 = 0
            for idx in sort_idx:
                beta_0 += np.log(1/self.betas[idx])
                if beta_0 > thres:
                    y_preds.append(h_T[idx, i])
                    break
        return np.array(y_preds)
                    