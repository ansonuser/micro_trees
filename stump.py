import numpy as np

class DecisionStump:
    
    def __init__(self):
        self.__parameter = None, None
        self.__mapping = None

    def train(self, X, Y, cuts):
        parameter = None, None # polar, f
        # X = self.encoding_feature(X, Y)
        
        cuts = np.quantile(X, [.25, .5, .75, 1.0])
        X = np.array(X)
        
        best_score = -1
       
        for c in cuts:
            for polar in [-1, 1]:
                prediction = np.zeros(len(Y))
                if polar == -1:
                    prediction[X<c] = 1
                else:
                    prediction[X>=c] = 1 
                acc = np.sum(prediction == Y)/len(Y)
                
                if acc > best_score:
                    best_score = acc
                    parameter = polar, c
                    
        self.__parameter = parameter
        print("best score = ", best_score)
        print("parameter:", self.__parameter)
        print("---------------------")
        print("Training end !")
    def show_models(self):  
        print(self.__parameter)
        
    def predict(self, X):
        assert self.__parameter[0] is not None
        prediction = np.zeros(len(X))
        if self.__parameter[0] == 1:
            prediction[X >= self.__parameter[1]] = 1
        else:
            prediction[X < self.__parameter[1]] = 1
        return prediction   
    
    
    def encoding_feature(self, X, Y):
        from collections import defaultdict
        y_counter = defaultdict(int)
        x_counter = defaultdict(int)
        
        for x,y in zip(X, Y) :
            if y == 1:
                y_counter[x] += 1
            x_counter[x] += 1
                
        keys = y_counter.keys()
        for k in keys:
            y_counter[k] = y_counter[k]/x_counter[k]
         
        f_list = []
        p_list = []   
        for k,v in y_counter.items():
            f_list.append(k)
            p_list.append(v)
            
      
        f_encodings = np.argsort(p_list)[::-1]
        
        mapping = {}
        for f,i in zip(f_list, f_encodings):
            mapping[f] = i.item()
        self.__mapping = mapping
        new_x = []
        
        for x in X:
            new_x.append(self.__mapping[x])
        print("mapping:", self.__mapping)
        return np.array(new_x)

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
    
