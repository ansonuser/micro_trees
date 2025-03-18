import numpy as np
from typing import List

class Node:
    """
    Each node records:
    
    Arguments:
        data: np.array
            The index of data which is in the node.
        
        feature_idx: idx
            Index of selected feature
            
        split_value: float
            The boundary of left and right subnodes
            
        side: str
            The node locates left or right of the parent
        
        layer: int
            The level of node.
    """
    def __init__(self, data, feature_idx, split_value, side, layer, parent=None):
        self.data = data
        self.feature_idx = feature_idx
        self.split_value = split_value
        self.left = None
        self.right = None
        self.layer = layer
        self.side = side
        self._prediction = None
        self.parent = parent

    @property
    def prediction(self):
        return self._prediction

    @prediction.setter
    def prediction(self, prediction):
        self._prediction = prediction
        
    def __repr__(self):
        if self.left is not None or self.right is not None:
            return "Node(layer=%2d, feature idx=%s, split value=%s, side=%s)"%(self.layer, str(self.feature_idx), str(self.split_value), str(self.side))
        else:
            return "Leaf(layer=%2d, feature idx=%s, split value=%s, side=%s)"%(self.layer, str(self.feature_idx), str(self.split_value), str(self.side))
        
        
        
def get_bin_indexes(X, cuts):
    n = len(cuts)
   
    bins = []
    for x in X:
        add = False
        for i in range(n-1):
            if x > cuts[i] and x <= cuts[i+1]:
                 bins.append(i+1)
                 add = True
                 break
        if not add:
            bins.append(0)
    return np.array(bins)


def get_dummies(Y:np.array):
    """
    Generate a dummies matrix
    
    Y: np.array
    """
    num_classes = len(set(Y))
    dummies = np.zeros((len(Y), num_classes))
    for i in range(num_classes):
        dummies[Y == i, i] = 1
    return dummies, num_classes
    
def multi_class_trainer(X, Y, classifer):
    """
    Use classifer train on X, Y_dummies[i] from i = 0, num_classes - 1
    and save all classifers in list to output
    
    --------------------------------
    Argurments:

        X: np.array
            Predictor
        
        Y_dummies: np.array
            A len(X) by num_classes dummies matrix
            
        num_classes: int
            Number of unique values in the target 
            
        classifer: DecisionTreeClassifier
    """
    Y_dummies, num_classes = get_dummies(Y)
    trees = []
    for i in range(num_classes):
        descision = classifer()
        descision.train(X, Y_dummies[:, i])
        trees.append(descision)
    return trees

def predictor(X:np.array, num_classes:int, models:List):
    """
    For inference purpose
    ---------------------------
    Arguments:
        X: np.array
            predictor
            
        num_classes : int
            Number of unique values in the target 
                
        models: List
            models[i] is for ith class prediction
            
        return: np.array
            return a len(X) by num_classes prediction
    """
    predictions = np.zeros((len(X), num_classes))
    for idx,tree in enumerate(models):
        predictions[:, idx] = tree.predict(X)
    return predictions