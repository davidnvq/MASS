import numpy as np 
import pandas as pd
from scipy.spatial.distance import *
from .liftTesc import LiftTesc
from scipy import sparse
from sklearn.metrics.pairwise import euclidean_distances

class MassClassifier(object):
    """MASS Multi-Label Classifier.
    Consume both labeled data and unlabeled data in training phase
    Use a semi-supervised clustering algorithm TESC.
    :param n_labels: declare the specific pre-defined of class labels.
    :type n_labels: int
    :param n_features: declare the number of features representing for a sample.
    :type n_features: int
    :param require_dense: whether the base classifier requires dense
        representations for input features and classes/labels matrices in fit/predict.
    :type require_dense: [bool, bool, bool]
    """
    def __init__(self, n_labels = 5, n_features=2300):
        self.n_labels = n_labels
        self.n_features = n_features
        self.call_fit = False
        #print("       MASS " + " n_labels =", self.n_labels,  ", n_features =", self.n_features)
        return
#########################################################################################


#########################################################################################
    def fit(self, X1, Y1, X2):
        """Fit classifier with training data    
        Parameters:
            :param X1: input features of labeled data
            :type X1:  sparse CSR matrix (n_samples, n_features)
            :param Y1: binary indicator matrix with label assignments of labeled data
            :type Y1:  dense matrix (n_samples, n_labels)
            :param X2: sparse CSR features of unlabeled data
            :type X2:  dense matrix (n_samples, n_features)
        Returns:
            Fitted instance of self
        """
        self.call_fit = True
        L1 = np.array([])
        L2 = np.array(range(0,self.n_labels))

        lt = LiftTesc(self.n_labels, self.n_features)
        lt.fit(X1, Y1, X2, L1, L2)

        self.cl_features = sparse.csr_matrix(np.empty((0, self.n_features), float))
        self.cl_labels = np.empty((0, self.n_labels), int)
        
        self.clusters = lt.get_clusters()
        for cluster in self.clusters:
            features = cluster.features
            labels = cluster.labels[1:].reshape(1, self.n_labels)
            self.cl_features = sparse.vstack([self.cl_features, features])
            self.cl_labels = np.append(self.cl_labels, labels, axis=0)
            
        self.export_clusters()
        return self

#########################################################################################
    def export_clusters(self, F_filename='models/cl_features.txt', L_filename='models/cl_labels.txt'):
        cl_features = np.array(self.cl_features.todense())
        cl_labels = np.array(self.cl_labels, dtype=int)
        np.savetxt(F_filename, cl_features, delimiter=' ')
        np.savetxt(L_filename, self.cl_labels, delimiter=' ')
        return
    
    def import_clusters(self, F_filename='models/cl_features.txt', L_filename='models/cl_labels.txt'):
        self.cl_features = np.loadtxt(F_filename, delimiter=' ')
        self.cl_features = sparse.csr_matrix(self.cl_features)
        self.cl_labels = np.loadtxt(L_filename, delimiter=' ')
        return
        
#########################################################################################
    def predict(self, X):
        """Predict a set of labels for X
        Internally this method uses a sparse CSR matrix representation for X
        
        Parameters:
            :param X: input features
            :type X:  sparse CSR matrix (n_samples, n_features)
        Returns:
            :returns: binary indicator matrix with label assignments
            :rtype:   dense matrix of int (n_samples, n_labels)
        """
        if self.call_fit == False: 
            self.import_clusters()
        indices = self._find_closest_clusters_indices(X)
        predicted_labels = self.cl_labels[indices]
        return(predicted_labels)
#########################################################################################


#########################################################################################
    def _find_closest_clusters_indices(self, X): 
        """Find the closest clusters for each samples in X
        Internally this method uses a sparse CSR matrix representation for X
        
        Parameters:
        :param X: input features. Each row of X is a vector representing a sample.
        :type X:  sparse CSR matrix (n_samples, n_features)
        
        Return:
        :returns: indices of closest clusters to the samples in X
        :rtype:   an index vector of int (n_samples)
        """
        dis_matrix = euclidean_distances(X, self.cl_features)
        #print(dis_matrix)
        indices = dis_matrix.argmin(axis=1)
        return(indices)

#########################################################################################



