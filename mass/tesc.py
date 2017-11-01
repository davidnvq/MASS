import numpy as np
import pandas as pd
from .cluster import *
from scipy.spatial.distance import *
from scipy import sparse
from sklearn.metrics.pairwise import euclidean_distances

class Tesc(object):
    """Tesc Classifier for single-label multi-class data
    Consume both labeled data and unlabeled data in training phase
    
    Paramters:
    :param n_labels: declare the specific pre-defined of class labels.
    :type n_labels: int
    :param n_features: declare the number of features representing for a sample.
    :type n_features: int
    """
    def __init__(self, n_labels=5, n_features=3):
        self.n_labels = n_labels
        self.n_features = n_features
        self.clusters = list()
        self.final_clusters = list()
        return

    def fit(self, X1, Y1, X2):
        """Fit classifier with training data    
        In TESC classifier, the assumed labels {lamdba1, lamdba2, lamdba3} are consumed instead of original labels.
        Parameters:
            :param X1: input features of labeled data
            :type X1:  sparse CSR matrix (n_samples, n_features)
            :param Y1: binary indicator matrix with label assignments of labeled data
            :type Y1:  dense matrix (n_samples, n_labels)
                       Only lambda label in Y1 is consumed in TESC
                       The original label set is mainly stored as information of clusters
            :param X2: input features of unlabeled data
            :type X2:  sparse CSR matrix (n_samples, n_features)
        Returns:
            Fitted instance of self
        """
        #print("-----TESC - fit.")
        self.X1 = X1
        self.Y1 = Y1
        self.X2 = X2
        self.cluster_features = sparse.vstack([self.X1, self.X2])
        self.run()
        return

    def run(self):
        """A procedure called in fit method of classifier which perfroms a set of sub-procedures.    
        A series of sub-procedures are call as below: 
            + Initialize_clusters
            + Perform clustering
        Parameters:
            :None:
        Returns:
            :None:
        """
        self.init_cluster()
        self.clustering()
        return

    def init_cluster(self):
        """	Initialize the clusters 
        Method:
            For each di of dataset D
	            Construct a cluster candidate Ci using each di and label Ci with Labels(di);
	            Set Ci as unidentified and the centroid of Ci as di;
	            Add Ci to cluster candidate set <clusters>;
        Parameters:
            :param D:   Include features X1, X2, and labels Y1.
            :type D:    Sparse CSR matrix for X1 and X2. Dense matrix for Y1
        Returns: 
            :param clusters: A set of initial clusters which have only one element.
            :type clusters:  A list []
        """
        #print("-----TESC - init_cluster!")
        for i in range(self.X1.get_shape()[0]):
            tmp = Cluster(n_features=self.n_features, n_labels=self.n_labels)
            tmp.init_centroid(self.X1.getrow(i), self.Y1[i])
            self.clusters.append(tmp)
        
        for i in range(self.X2.get_shape()[0]):
            tmp = Cluster(n_features=self.n_features, n_labels=self.n_labels)
            tmp.init_centroid(self.X2.getrow(i))
            self.clusters.append(tmp)
        return

    def clustering(self):
        """	Clustering method
        Method:
            Find 2 closest clusters in the list. 
            Determine whether we should merge 2 clusters or mark them defined.
        Parameters:
            :param clusters: A set of initial clusters which have only one element.   
            :type clusters:  A list []
        Returns: 
            :param clusters: A set of final clusters.
            :rtype: A list []
        """
        #print("-----TESC - clustering.")    
        n_iters = 0
        while True:
            n_iters += 1
            #print("Loop", n_iters, "on TESC")
            #print("No of clusters", len(self.clusters))
            index1, index2 = self.find_min_distance_indices()
            self.examine_clusters(index1, index2)      
            if self.get_n_undefined_clusters() < 2:
                break
            if n_iters > 1000000:
                break
        return

    
    def get_n_undefined_clusters(self):
        """ Get the number of undefined clusters method
        Parameters:
            {None}
        Returns: 
            :returns: the number of undefined clusters.
            :rtypes: int
        """    
        #print("-----TESC - get the number of undefined clusters!")
        n_undefined = 0
        for cl in self.clusters:
            if cl.defined == False:
                n_undefined += 1
        return n_undefined

    def find_min_distance_indices(self):
        """ Find 2 closest clusters method
        Parameters:
            {None}
        Returns: 
            :returns: the indices of the two closest clusters in the list of clusters.
            :rtypes: tuple (int, int)
        """    
        #print("-----TESC - find_min_distance_indices!")
        temp = euclidean_distances(self.cluster_features, self.cluster_features)
        np.fill_diagonal(temp, np.Inf)
        rows, cols = np.where(temp == temp.min())
        return (rows[0], cols[0])
    
    def replace_row(self, A, B, index):
        """ Replace the index(th) row in a sparse CSR matrix with another row
        Parameters:
            :param A:       The matrix we need to replace a row index(th)
            :type A:        A sparse CSR matrix
            :param B:       An array will replace the index(th) row in A
            :type B:        A numpy array []
            :param index:   the index of row in A that will be replaced
            :type index:    int
        Returns: 
            :returns: A replaced matrix
            :rtype:   A sparse CSR matrix  
        """    
        Atemp = sparse.coo_matrix(A[index])
        Al = A.tolil()
        Bl = sparse.coo_matrix(B)
    
        for row, col, val in zip(Atemp.row, Atemp.col, Atemp.data):
            Al[index, col] = 0
        for row, col, val in zip(Bl.row, Bl.col, Bl.data):
            Al[index, col] = val
        return sparse.csr_matrix(Al)
    
    def delete_rows(self, A, indices):
        """ Delete a list of rows in a sparse CSR matrix
        Parameters:
            :param A:       The matrix we need to delete some rows
            :type A:        A sparse CSR matrix
            :param index:   the list of indices of rows in A that will be deleted
            :type indices:  a list []
        Returns: 
            :returns: A new modified matrix
            :rtype:   A sparse CSR matrix  
        """    
        Al = A.tolil()
        preserved_rows = [i for i in range(Al.shape[0]) if i not in indices]
        Al = A[preserved_rows, :]
        return sparse.csr_matrix(Al)

    def examine_clusters(self, index1, index2):
        """Determine what to do next when finding out two closest clusters.
        Parameters:
            :param index1: The index of the first cluster
            :type index1:  int
            :param index2: The index of the second cluster
            :type index2:  int
        Returns: 
            {None}
        """    
        #print("-----TESC - examine_clusters!")
        lamda1 = self.clusters[index1].get_lamda()
        lamda2 = self.clusters[index2].get_lamda()
        
        # If 2 clusters' lambda labels are similar
        if lamda1 == lamda2:
            #print("TH1")
            self.clusters[index1].merge_cluster(self.clusters[index2])
            # Update new feature vector and remove old value
            self.cluster_features = self.replace_row(self.cluster_features, self.clusters[index1].features, index1)
            self.cluster_features = self.delete_rows(self.cluster_features, [index2])
            self.clusters.pop(index2)
    
    
        
        # If the 1st cluster is unlabeled, the 2nd cluster is labeled
        elif lamda1 != 0 and lamda2 == 0:
            #print("TH2")
            self.clusters[index1].merge_cluster(self.clusters[index2])
            self.cluster_features = self.replace_row(self.cluster_features, self.clusters[index1].features, index1)
            self.cluster_features = self.delete_rows(self.cluster_features, [index2])  
            self.clusters.pop(index2)
             
        # If the 1st cluster is labeled, the 2nd is unlabled
        elif lamda1 == 0 and lamda2 != 0:
            #print("TH3")
            self.clusters[index2].merge_cluster(self.clusters[index1])
            self.cluster_features = self.replace_row(self.cluster_features, self.clusters[index2].features, index2)
            self.cluster_features = self.delete_rows(self.cluster_features, [index1])
            self.clusters.pop(index1)
            
        # If 2 clusters' lambda labels are different
        else:
            #print("TH4")
            self.clusters[index1].defined = True
            self.clusters[index2].defined = True
            tmp1 = self.clusters[index1]
            tmp2 = self.clusters[index2]
            self.final_clusters.append(tmp1)
            self.final_clusters.append(tmp2)
            # Remove clusters from current clusters
            # http://stackoverflow.com/questions/11303225/how-to-remove-multiple-indexes-from-a-list-at-the-same-time
            remove_indices = [index1, index2]
            self.clusters = [i for j, i in enumerate(self.clusters) if j not in remove_indices]
            # Remove a list of rows from np array
            # https://docs.scipy.org/doc/numpy-1.10.4/reference/generated/numpy.delete.html
            self.cluster_features = self.delete_rows(self.cluster_features, [index1, index2])  

        return

    def get_clusters(self):
        """ Return a list of clusters
        Parameters:
            {None}
        Returns: 
            :returns: A list of clusters
            :rtype:   A list []
        """    
        self.final_clusters.extend(self.clusters)
        return self.final_clusters

if __name__ == "__main__":
    print("Hello, it's from main!")
    label_data = pd.read_csv("label.csv")
    unlabel_data = pd.read_csv("unlabel.csv")
    X1 = label_data.iloc[:,5:]
    Y1 = label_data.iloc[:,:5]
    X2 = unlabel_data
    tesc = Tesc()
    tesc.fit(X1, Y1, X2)
