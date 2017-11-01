import numpy as np 
import pandas as pd 
from scipy import sparse

class Cluster(object):
    """ A Cluster which is a group of samples with high similarities
        A cluster is defined by: 
            a centroid: - A set of labels representing for the cluster
                        - A set of features which is the mean of all samples in this cluster.
            a list of samples: which belong to this cluster.
            
    Parameters:
        :param n_labels: the number of pre-defined labels of labeled data
        :type n_labels: int
        :param n_labels: the number of pre-defined labels of labeled data
        :type n_labels: int
    """
    def __init__(self, n_labels=5, n_features=3):
        self.defined = False
        self.n_labels = n_labels
        self.n_features = n_features
        self.size = 0
        self.X2 = sparse.csr_matrix(np.empty((0, self.n_features), float))
        self.X1 = sparse.csr_matrix(np.empty((0, self.n_features), float))
        self.Y1 = np.empty((0, self.n_labels + 1), int)
    
        self.labels = np.empty((0, self.n_labels + 1), int)
        self.features = np.empty((0,self.n_features), float)
    
    # Initialize a cluster from a labeled sample or unlabeled sample
    def init_centroid(self, X, Y=None):
        """Initialize the centroid of cluster    
        Parameters:
            :param X: input features of labeled data or unlabeled data
            :type X:  sparse CSR matrix (n_samples, n_features)
            :param Y: binary indicator matrix with label assignments of labeled data (if any)
            :type Y:  dense matrix (n_samples, n_labels)
        Returns:
            initialize the centroid for this cluster
        """
        self.size = 1
        self.features = X
        if not (Y is None):
            self.labels = Y
            self.X1 = X
            self.Y1 = np.array(Y).reshape(1, self.n_labels + 1)
            
        else:
            self.X2 = X

    def toString(self):
        cl_str = "---- Cluster ----" + "\n" 
        cl_str += "cluster size = " + str(self.size) + "\n" 
        cl_str += "labels = " + str(self.labels) + "\n" 
        cl_str += "centroid = " + str(self.features) + "\n" 
        cl_str += "X" + "\n" 
        cl_str += str(sparse.vstack([self.X1, self.X2])) + "\n" 
        cl_str += "Y" + "\n" 
        cl_str += str(self.Y1) + "\n" 
        cl_str += "---- End ----" + "\n" 
        return cl_str
    
    # Append X1, Y1, X2
    def merge_cluster(self, ano_cluster):
        """ A method to merge the current cluster with another cluster
        
        Parameters:
            :param ano_cluster: Another cluster we need to merge with current cluster
            :param ano_cluster: An object of <Cluster>
        Returns:
            :returns:   An updated cluster
            :rtypes:    <Cluster>
        """
        #print("----cluster: Merge cluster")
        # Append labeled data and unlabeled data
        self.X1 = sparse.vstack([self.X1, ano_cluster.X1])
        self.X2 = sparse.vstack([self.X2, ano_cluster.X2])
        self.Y1 = np.append(self.Y1, ano_cluster.Y1, axis=0)
        
        # Update centroid and size        
        if self.features.size == 0:
            self.features = ano_cluster.features
            self.labels = ano_cluster.labels
        else:
            self.features = (self.features * self.size + ano_cluster.features * ano_cluster.size) / (self.size + ano_cluster.size)
        
        self.size = self.size + ano_cluster.size
        return
    

    def check_label_similarity(self):
        """ A method checks whether all the samples in the cluster have the same labels or not
        See this link http://stackoverflow.com/questions/14859458/how-to-check-if-all-values-in-the-columns-of-a-numpy-matrix-are-the-same
        A row of  Y = {lamda, label_1, label_2, label_3, label_nlabels}
        Parameters:
            {None}
        Returns:
            :returns:   Whether the condition satifies
            :rtypes:    boolean {True, False}
        """
        #print("----cluster: Check_label_similarity")
        if self.Y1.size == 0:
            return False
        labels = self.Y1[:, 1:]
        temp = np.all(labels == labels[0,:], axis=0)
        return (np.sum(temp) == self.n_labels)


    
    def check_empty(self):
        """ A method checks whether the cluster is empty or not
        Parameters:
            {None}
        Returns:
            :returns:   Whether the condition satifies
            :rtypes:    boolean {True, False}
        """
        return (self.size == 0) 
    
    def get_lamda(self):
        """ A get lamdba method
        Parameters:
            {None}
        Returns:
            :returns:   lambda
            :rtypes:    int
        """
        if self.labels.size == 0:
            return 0
        else: 
            return self.labels[0]
    

# Debug
if __name__ == "__main__":
    print("hello Cluster!")    
    a = Cluster(np.array([[1,2,3]]), n_labels=1, n_features=3)
    print(a.X1)
    print(a.X2)
    print(a.Y1)
    print(a.size)
    print("empty = ", a.check_empty())
    print("similarity = ", a.check_label_similarity())

    b = Cluster(np.array([[9,9,9],[10,10,10]]), np.array([[3, 2],[1, 2]]), n_labels=1, n_features=3)
    b.merge_cluster(a)
    print("After merging!")
    print(b.X1)
    print(b.X2)
    print(b.Y1)
    print(b.size)
    print(b.check_empty())
    print(b.check_label_similarity())



