import numpy as np
import pandas as pd 
import os 
import sys 
from .tesc import *

class LiftTesc(object): 
    """LiftTesc Class
    Consume both labeled data and unlabeled data in training phase
    Take the ideas of dividing training dataset into subspaces and pass it to TESC.
    
    Paramters:
    :param n_labels: declare the specific pre-defined of class labels.
    :type n_labels: int
    :param n_features: declare the number of features representing for a sample.
    :type n_features: int
    """
    def __init__(self, n_labels=5, n_features=3):
        self.n_labels = n_labels
        self.n_features = n_features
        self.final_clusters = list()
        return

    def fit(self, X1, Y1, X2, L1, L2):
        """Fit classifier with training data    
        Parameters:
            :param X1: input features of labeled data
            :type X1:  sparse CSR matrix (n_samples, n_features)
            :param Y1: binary indicator matrix with label assignments of labeled data
            :type Y1:  dense matrix (n_samples, n_labels)
            :param X2: input features of unlabeled data
            :type X2:  sparse CSR matrix (n_samples, n_features)
            :param L1: A set of labels that all appear in every row of Y1
            :type L1:  A array of int. for example [1, 3]
            :param L2: A set of labels that compliment L1. L2 U L1 = L
            :type L2:  A array of int. for example [2, 4, 5]
        Returns:
            Fitted instance of self
        """
       #print("LIFT_TESC - fit!")
        self.X1 = X1
        self.Y1 = Y1
        self.Label_lamda = np.zeros((Y1.shape[0], 1), dtype=int) #Only labeled data need lamda
        self.X2 = X2
        self.L1 = L1
        self.L2 = L2
        self.run()
        return self

    def run(self):
        """A procedure called in fit method of classifier which perfroms a set of sub-procedures.    
                A series of sub-procedures are call as below: 
                    find_lambda, assign_DL_with_lambda, run_TESC_on_D
                    get_clusters_from_TESC,
                    clustering_on_D1, clustering_on_D2, clustering_on_D3
                    print_clusters
        Parameters:
            :None:
        Returns:
            :None:
        """
        if (self.L1.size == 0):
            self.find_lambda_greedy()
        else: 
            self.find_lambda_graph()
        
        self.assign_DL_with_lambda()
        self.run_Tesc_on_D()
        self.get_clusters_from_Tesc()
        self.clustering_on_D1();
        self.clustering_on_D2();
        self.clustering_on_D3();
        #self.print_clusters();
        return
    
    def find_lambda_graph(self):
        """Find the lamda in L2 which make a co-concurrent graph with L1
            If there are more than 1 labels with the same maximum, 
            Choose the label with the lowest index.
        Parameters: 
            :param Y1:  the label set of labeled data
            :type Y1:   dense matrix (n_samples, n_labels)
            :param L2:  L2 = L/L1 Example, L2 = {a, b, c, e, f}
         
        Returns:
            :returns: The index of lmbda {b}
            :rtype:   int
        """
        #print("LIFT_TESC - find lambda - Co-occurence Graph!")
        for label in self.L2:
            max_value = -1
            max_index = -1
            val_label = 0
            for y in self.Y1:
                for l1 in self.L1:
                    l1 = int(l1)
                    if (y[label] == 1 and y[l1] == 1):
                        val_label += 1
            if (max_value < val_label):
                max_value = val_label
                max_index = label
        self.lamda = max_index
        return

    def find_lambda_greedy(self):
        """Find the lamda appears in Y1 (the label of L2 appears most frequent)
            If there are more than 1 labels with the same maximum, 
            Choose the label with the lowest index.
        Parameters: 
            :param Y1:  the label set of labeled data
            :type Y1:   dense matrix (n_samples, n_labels)
            :param L2:  L2 = L/L1 Example, L2 = {a, b, c, e, f}
         
        Returns:
            :returns: The index of lmbda {b}
            :rtype:   int
        """
        #print("LIFT_TESC - find lambda - Greedy!")
        label_sum = np.sum(self.Y1, axis=0)
        if label_sum.ndim == 2:
            label_sum = label_sum.getA1()
        max_index = -1
        max_value = -1
        for i in range(self.n_labels):
            if max_value < label_sum[i] and np.any(self.L2 == i): #Check lamda in L2 or not
                max_value = label_sum[i]
                max_index = i
        self.lamda = max_index
        return
    
    def assign_DL_with_lambda(self):
        """Assign the labeled data with assumed labels - lamdba1, lamdba2, lamdba3.
        Method: 
            If labeled data only consists of L1 and lamdba.
                Assign Label Lamdba = 1
            If labeled data only consists of L1 and lamdba and other labels.
                Assign Label Lamdba = 2
            If labeled data only doesnt contain lamdba.
                Assign Label Lamdba = 3
        Parameters: 
            :lamda:        The lamda found from the set L2 satifying the condition.
            :type lamda:   int
        Returns:
            :returns:      A list of lamdba labels for labeled data, attached it to the list of Y1.
            :rtype:        A dense matrix Y1_lambda
        """
        #print("LIFT_TESC - assign DL with Lambda!")

        #Add lambda to L1 and start classifying
        L1_lamda = np.append(self.L1, self.lamda)
        L1_lamda = L1_lamda.astype(int)
        #Make a list of ~L1_lambda by using mask
        mask = np.ones(self.n_labels, dtype=bool)
        mask[L1_lamda] = False

        n_elements = self.Y1.shape[0]
        for i in range(n_elements):
            cond1 = not np.any(self.Y1[i, L1_lamda] == 0)
            cond2 = not np.any(self.Y1[i, mask] == 1)

            if cond1 and cond2:
                self.Label_lamda[i] = 1              # Lamda1
            elif cond1 and not cond2:
                self.Label_lamda[i] = 2              # Lamda2
            else:
                self.Label_lamda[i] = 3              # Lamda3
        self.Y1 = np.append(self.Label_lamda, self.Y1, axis=1)
        return

    def run_Tesc_on_D(self):
        """The procedure performing TESC on training data.
        Parameters:
            {none}
        Returns:
            {none}
        """
        #print("LIFT_TESC - run Tesc on D!")
        self.tesc = Tesc(n_features=self.n_features, n_labels=self.n_labels)
        self.tesc.fit(self.X1, np.array(self.Y1), self.X2)
        return

    def get_clusters_from_Tesc(self):
        """The procedure dividing D into D1, D2, and D3 as result from TESC clustering.
            Method: 
            If labeled data only consists of L1 and lamdba.
                Assign data to D1
            If labeled data only consists of L1 and lamdba and other labels.
                Assign data to D2
            If labeled data only doesnt contain lamdba.
                Assign data to D3
        Parameters: 
            {none}
        Returns:
            {none}
        """
        #print("LIFT_TESC - get clusters from Tesc!")
        self.D1 = Cluster(n_labels=self.n_labels, n_features=self.n_features)
        self.D2 = Cluster(n_labels=self.n_labels, n_features=self.n_features)
        self.D3 = Cluster(n_labels=self.n_labels, n_features=self.n_features)

        n_elements = 0 
        clusters = self.tesc.get_clusters()
        for cl in clusters:
            lamda = cl.get_lamda()
            if (lamda == 1):
                #print("m1")
                self.D1.merge_cluster(cl)
            if (lamda == 2):
                #print("m2")
                self.D2.merge_cluster(cl)
            if (lamda == 3):
                #print("m3")
                self.D3.merge_cluster(cl)
        return 

    def clustering_on_D1(self):
        """Perform clustering on D1
        Method: 
            Add D1 into the set of output clusters
        Parameters: 
            :param D1:     A cluster of labeled data and unlabeled data which are assigned to lambda1.
            :type D1:      An object of <class> Cluster
        Returns:
            :returns:      A list of output clusters
            :rtype:        A list [].
        """
        #print("LIFT_TESC - clustering on D1!")
        if self.D1.check_empty() != True:
            if self.D1.check_label_similarity() == True:
                self.final_clusters.append(self.D1)
            else:
                print("---------Error at adding D1 to clusters")
        return 

    def clustering_on_D2(self):
        """Perform clustering on D2
        Method: 
            Add D2 to a set of output clusters if all data in D2 contains the same labels.
            Otherwise: 
                Call LiftTesc on D2
        Parameters: 
            :param D2:     A cluster of labeled data and unlabeled data which are assigned to lambda2.
            :type D2:      An object of <class> Cluster
        Returns:
            :returns:      A list of output clusters
            :rtype:        A list [].
        """
        #print("LIFT_TESC - clustering on D2!")
        if self.D2.check_empty() != True:
            if self.D2.check_label_similarity() == True:
                self.final_clusters.append(self.D2)
            else:
                lt = LiftTesc(n_labels=self.n_labels, n_features=self.n_features)
                lt.fit(self.D2.X1, self.D2.Y1[:, 1:], self.D2.X2, np.append(self.L1, self.lamda), self.L2[self.L2 != self.lamda])
                self.final_clusters.extend(lt.get_clusters())
        return 

    def clustering_on_D3(self):
        """Perform clustering on D3
        Method: 
            Add D3 to a set of output clusters if all data in D3 contains the same labels.
            Otherwise: 
                Call LiftTesc on D3
        Parameters: 
            :param D3:     A cluster of labeled data and unlabeled data which are assigned to lambda3.
            :type D3:      An object of <class> Cluster
        Returns:
            :returns:      A list of output clusters
            :rtype:        A list [].
        """
        #print("LIFT_TESC - clustering on D3!")
        if self.D3.check_empty() != True:
            if self.D3.check_label_similarity() == True:
                self.final_clusters.append(self.D3)
            else:
                lt = LiftTesc(n_labels=self.n_labels, n_features=self.n_features)
                lt.fit(self.D3.X1, self.D3.Y1[:, 1:], self.D3.X2, self.L1, self.L2[self.L2 != self.lamda])
                self.final_clusters.extend(lt.get_clusters())
        return 

    def get_clusters(self):
        """Get output clusters method
        Parameters: 
            {none}
        Returns:
            :returns:      A list of output clusters
            :rtype:        A list [].
        """
        return self.final_clusters

    def print_clusters(self):
        """Print output clusters method
            Print the information of clusters in the output list
        Parameters: 
            {none}
        Returns:
            {none}
        """
        text_file = open("Output.txt", "w")
        myStr = ""
        for cl in self.final_clusters:
            myStr += cl.toString() + "\n"
        text_file.write(myStr)
        text_file.close()
        return 

#DEBUG
if __name__ == "__main__":
    lt = LiftTesc(n_labels=5)
    label_data = pd.read_csv("label.csv")
    unlabel_data = pd.read_csv("unlabel.csv")
    X1 = label_data.iloc[:,5:]
    Y1 = label_data.iloc[:,:5]
    X2 = unlabel_data
    X1 = np.array(X1)
    Y1 = np.array(Y1)
    X2 = np.array(X2)
    L1 = np.array([], dtype=int)
    L2 = np.array([0,1,2,3,4], dtype=int)
    lt.fit(X1, Y1, X2, L1, L2)
    lt.print_clusters()
    

        



