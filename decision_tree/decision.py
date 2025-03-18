from collections import defaultdict
import numpy as np
import random
import graphviz
import sys
import os
sys.path.append(os.getcwd() + f"{os.sep}..")
from utils.helper import Node
random.seed(2025)
np.random.seed(2025)





class DecisionTreeRegressor:
    def __init__(self):
       
        self.root = None
        self.used = defaultdict(list)
        # self.impurity = 1e10
   
    def predict(self, X):
        assert self.root is not None
        pointer = self.root
        
        def trace(pointer, x):
            # prediction = pointer.prediction
            while pointer:
                prediction = pointer.prediction
                pointer = pointer.left if x[pointer.feature_idx] < pointer.split_value else pointer.right             
            return prediction
        
        predictions = []      
        for x in X:
            out = trace(pointer, x)
            predictions.append(out)
        return np.array(predictions)
    
    def friedman_mse(self, y_left, y_right):
        """
        Compute the Friedman MSE criterion for a potential split.
    
        Parameters:
        y_left : (np.array)
            Target values in the left child node.
            
        y_right :(np.array): 
            Target values in the right child node.
    
        Returns: float
            The Friedman MSE value for the split.
        """
        n_L, n_R = len(y_left), len(y_right)
        n_T = n_L + n_R 
    
        if n_L == 0 or n_R == 0:  # Avoid division by zero
            return float('inf')
    
        # Compute variances
        sigma_L2 = np.var(y_left, ddof=1) if n_L > 1 else 0  
        sigma_R2 = np.var(y_right, ddof=1) if n_R > 1 else 0  
    
        # Friedman MSE calculation
        mse_friedman = (n_L * n_R / n_T) * ((sigma_L2 / n_L) + (sigma_R2 / n_R))
        
        return mse_friedman
       
    def train(self, X, Y, minimum_sample=4, maximum_depth=4):
        n,m = X.shape
        
        def build(node):
            if node is None or node.layer == maximum_depth:
                return
                
            index = node.data
            
            feature_idx = None
            thres = None
            best_score = 1e10
            for i in range(m):
                split_thres = np.median(X[index, i])
                indexer1 = X[index, i] < split_thres 
                s = indexer1.sum()
                if s < 2 or s > len(indexer1) - 2:
                    continue
                friedman_score = self.friedman_mse(Y[index][indexer1], Y[index][~indexer1])
                if friedman_score < best_score:
                    best_score = friedman_score
                    feature_idx = i
                    left_index = index[indexer1]
                    right_index = index[~indexer1]
                    thres = split_thres
     
            if feature_idx is not None and thres is not None:
                if thres not in self.used[feature_idx]:
                    self.used[feature_idx].append(thres)
                    
                    if self.root is None:
                        self.root = node
                        node.feature_idx = feature_idx
                        node.split_value = thres
                
                    if len(left_index) > minimum_sample :
                        node.left = Node(left_index, feature_idx, thres, "left", node.layer+1)
                        node.left.prediction = np.median(Y[left_index])
                    if len(right_index) > minimum_sample :
                        node.right = Node(right_index, feature_idx, thres, "right", node.layer+1)
                        node.right.prediction = np.median(Y[right_index])
                    build(node.left )
                    build(node.right)
        init_node = Node(np.arange(n), None, None, None, 0)
        init_node.prediction = np.median(Y)
        build(init_node)         
    
class DecisionTreeClassifier:
    def __init__(self, eps=1e-8):
        self.root = None
        self.used = defaultdict(list)
        self.leaves = []
        self.epsilon = eps
        
    def showtree(self):
        g = graphviz.Digraph(name='decision tree', format="png", graph_attr={ "size":"8,5"})
        pointer = self.root
        def expand(pointer):
            if pointer is None:
                return
            g.node(str(pointer))
            if pointer.left is not None:
                g.edge(str(pointer), str(pointer.left), label="Yes")
                expand(pointer.left)
    
            if pointer.right is not None:
                g.edge(str(pointer), str(pointer.right), label="No")
                expand(pointer.right)
        expand(pointer)
        return g

    # def predict_prob(self, X):
    #     out = self.predict(X)
    #     out = np.exp(out)
    #     denom = out.sum(axis=1, keepdims=True)+self.epsilon
    #     return out/denom

    def train(self, X, Y, maximum_depth=3, minimum_sample=2):
        n,m = X.shape
        
        def build(node):
            if node is None :
                return 

            if node.layer == maximum_depth:
                self.leaves.append(node)
                return
     
            index = node.data  
            feature_idx = None
            thres = None

            best_score = 1e10
            for i in range(m):
                for split_thres in X[index, i]:
                    indexer1 = X[index, i] < split_thres 
                    s = indexer1.sum()
                    if 1 < s < (len(index)-1):
                        score = self.impurity(Y[index][indexer1]) + self.impurity(Y[index][~indexer1])
                        if score < best_score:
                            best_score = score
                            feature_idx = i
                            left_index = index[indexer1]
                            right_index = index[~indexer1]
                            thres = split_thres

            if feature_idx is not None and thres is not None:
                if thres not in self.used[feature_idx]:
                    self.used[feature_idx].append(thres)
                    if self.root is None:
                        node.feature_idx = feature_idx
                        node.split_value = thres
                        self.root = node
                
                    if len(left_index) > minimum_sample and len(right_index) > minimum_sample:
                        node.left = Node(left_index, feature_idx, thres, "left", node.layer+1, node)
                        node.left.prediction = Y[left_index].mean(axis=0)
                        node.right = Node(right_index, feature_idx, thres, "right", node.layer+1, node)
                        node.right.prediction = Y[right_index].mean(axis=0)
                        build(node.left )
                        build(node.right)
                    else:
                        self.leaves.append(node)
                        return 
                   
        init_node = Node(np.arange(n), None, None, None, 0, None)
        init_node.prediction = Y.mean(axis=0)
        build(init_node) 
        
    def impurity(self, Y):
        return np.std(Y, axis=0).sum()

    def predict(self, X):
        assert self.root is not None
        pointer = self.root
        
        def trace(pointer, x):
            while pointer:
                prediction = pointer.prediction
                pointer = pointer.left if x[pointer.feature_idx] < pointer.split_value else pointer.right             
            return prediction
        
        predictions = []      
        for x in X:
            out = trace(pointer, x)
            predictions.append(out)
        
        return np.r_[predictions]

