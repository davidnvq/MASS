import numpy as np
import pandas as pd

class DimensionReduction(Object):

    def __init__(n_dims=1200):
        self.n_dims = n_dims
        
    def fit_transform(filename, X):
        self.fit(filename)
        return self.transform(X)

    def fit(filename="/models/ranklist.txt"):
        self.ranklist = np.loadtxt(filename, delimiter=",")
        return 

    def transform(X):
        return self.X[:, ranklist[:n_dims]]
    
