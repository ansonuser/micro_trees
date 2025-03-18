import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.getcwd() + "\\..")
from decision_tree.decision import DecisionTreeClassifier

class GradientBoostClassifier:
    """
    Simplified version of gradientboost classifier.
    
    --------------------
    Arguments:
        n_step: int
            Number of trees 
            
        learning_rate: float
            Coefficient of weight update
            
        epsilon: float
            A very small number to avoid error
            
        early_stop: int
            Early stop if doesn't improve in given steps.
            
        max_depth: int
            The deepest level it can go.
    """
    def __init__(self, n_step:int, learning_rate:float, num_class:int, epsilon:float, early_stop:int, max_depth:int=1):
        self.n_step = n_step
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.num_class = num_class
        self.boost = []
        self.loss = []
        self.early_stop = early_stop
        self.max_depth = max_depth
     
    def train(self, X, Y):
        """
        Model trained with input (X, Y)
        -------------------
        Arguments:
            X : np.array
                N by K predictor matrix where K is number of features
            Y : np.array
                N by 1 prediction vector
        """
        Y_ohe = pd.get_dummies(Y).values.astype("float")
        R = Y_ohe.copy()
        prediction = np.zeros((len(X), self.num_class))
        Prob = self._softmax(prediction)
        
        best_loss = 1e10
        update_step = 0
        for i in range(self.n_step):
            trees = []
            for k in range(self.num_class):
                tree = DecisionTreeClassifier(eps=self.epsilon)
                tree.train(X, R[:, k], maximum_depth=self.max_depth)
                Prob = self._softmax(prediction)
                ng = self._neg_gradient(Y_ohe[:, k], Prob[:, k])
                hessian = self._hessian(Prob[:, k])
                for leaf in tree.leaves:
                    leaf.prediction = ng[leaf.data].sum()/(hessian[leaf.data].sum()+self.epsilon)
                prediction[:, k] += self.learning_rate*tree.predict(X)
                trees.append(tree)
                R[:,k] = ng
            
            loss = self.entropy(Prob, Y_ohe)
            
            self.loss.append(loss)
            self.boost.append(trees)
            if loss < best_loss:
                best_loss = loss
                update_step = i
            if  i - update_step > self.early_stop:
                print("Early stop at step={}".format(i))
                break
        
    def train_multi(self, X, Y):
        """
        Multi-Label version of "train"
        """
        Y_ohe = pd.get_dummies(Y).values.astype("float")
        R = Y_ohe.copy()
        prediction = np.zeros((len(X), self.num_class))
        for i in range(self.n_step):   
            tree = DecisionTreeClassifier(self.epsilon)
            tree.train(X, R, maximum_depth=self.max_depth)
            prediction += self.learning_rate*tree.predict(X)
            Prob = self._softmax(prediction)
            
            self.loss.append(self.entropy(Prob, Y_ohe))
            ng = self._neg_gradient(Y_ohe, Prob)
            hessian = self._hessian(Prob)
            for leaf in tree.leaves:
                for j in range(self.num_class):
                    leaf.prediction[j] = ng[leaf.data, j].sum()/(hessian[leaf.data, j].sum()+self.epsilon)

            self.boost.append(tree)
            R = ng
            
    def entropy(self, prediction, target):
        return -(target*np.log(prediction+self.epsilon)).sum().sum()/len(target)

    def predict_prob(self, X):
        """
        predict probability on X
        
        return: np.array
            probability  of each class
        """
        prediction = np.zeros((len(X), self.num_class))
        for trees in self.boost:
            for k,tree_k in enumerate(trees):
                prediction[:, k] +=  self.learning_rate*tree_k.predict(X)
        return self._softmax(prediction)
    
    def predict_prob_multi(self, X):
        out = np.zeros((len(X), self.num_class))
        out = out + self.learning_rate * sum(tree.predict(X) for tree in self.boost)        
        return self._softmax(out)

    def _softmax(self, x):
        x -= x.max(axis=1, keepdims=True)
        log_sum_exp = np.log(np.exp(x).sum(axis=1, keepdims=True) + self.epsilon)  # Log-sum-exp
        return np.exp(x - log_sum_exp)
        

    def predict(self, X, multi=False):
        if multi:
            out = self.predict_prob_multi(X)
        else:
            out = self.predict_prob(X)
        return np.argmax(out, axis=1)
            
        
    @classmethod 
    def _hessian(cls, prob):
        return prob*(1-prob)

    @classmethod
    def _neg_gradient(cls, y, prob):
        return y - prob