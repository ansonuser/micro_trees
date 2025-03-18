import random
import sys
import os
sys.path.append(os.getcwd() + f"{os.sep}..")
from decision_tree.decision import DecisionTreeClassifier
from utils.helper import get_dummies, multi_class_trainer, predictor
import numpy as np

class RandomForest:
    """
    A simplifed version of random forest
    
    -----------------------------
    Arguments:
        num_of_tree: int
            Number of trees 
            
        fraction_of_size: float
            Ratio of data used per tree.
            
        fraction_of_feature: float
            Ratio of number of features used per step.
            
        num_of_classes: int
            Number of unique value in target
    """
    def __init__(self, num_of_tree, fraction_of_size, fraction_of_feature, num_of_classes=None):
        self.num_of_tree = num_of_tree
        self.fraction_of_size = fraction_of_size
        self.fraction_of_feature = fraction_of_feature
        self.trees = []
        self.num_of_classes = num_of_classes
    def train(self, X, Y, minimum_sample=3, max_depth=3):
        Y_dummies, num_of_classes = get_dummies(Y)
        n,m = X.shape
        for _ in range(self.num_of_tree):
            select_idx = random.sample(range(n), int(n*self.fraction_of_size))
            used_fe_idx = random.sample(range(m), int(m*self.fraction_of_feature))
            used_fe_idx.sort()
            sub_x = X[select_idx]
            cur_model = []
            for c in range(num_of_classes):
                sub_y = Y_dummies[select_idx, c]
                descision = DecisionTreeClassifier()
                descision.train(sub_x, sub_y, minimum_sample, max_depth)
                assert descision.root is not None
                cur_model.append(descision)
            self.trees.append(cur_model)

    def predict_prob(self, X):
        k = len(self.trees[0])
        out = np.zeros((len(X), k))
        n_trees = float(len(self.trees))
        for model in self.trees:
            for i, model_i in enumerate(model):
                assert model_i is not None
                out[:, i] += model_i.predict(X)
        out /= n_trees
        return out
            