"""
Run MASS on T4 with the support of knowledge from T1
T1 = data 26800 
T4 = our data
"""

import numpy as np 
import pandas as pd
from mass.mc import MassClassifier

from scipy import sparse
from eval import Evaluation


mass_config = [{"name": "MASS", "classifier" : MassClassifier}]

T4_name = ['a', 'b', 'c', 'd', 'e', 'f']
T4_size = [100, 200, 400, 600, 1000, 1200]

class Classifier(object):
    
    def __init__(self, conf):
        print("       0. Init " + conf["name"] + " classifier\n")
        self.conf = conf
        return

    def fit(self, X_train, Y_train, X_unlabel=None):
        print("       1. Fit " + self.conf["name"] + " classifier")

        if self.conf["name"] != "MASS":
            self.classifier = self.conf["classifier"]()
            Y_train = sparse.csr_matrix(Y_train)
            self.classifier.fit(X_train, Y_train)

        else:
            X_train = sparse.csr_matrix(X_train)
            X_unlabel = sparse.csr_matrix(X_unlabel)
            
            self.classifier = self.conf["classifier"](n_features=X_train.get_shape()[1],n_labels=Y_train.shape[1])
            self.classifier.fit(X_train, Y_train, X_unlabel)
        return self
    
    def predict(self, X_test):
        print("       2. Predict Y_pred for X_test")
        X_test = sparse.csr_matrix(X_test)           
        Y_pred = self.classifier.predict(X_test)
        if self.conf["name"] != "MASS":
            Y_pred = np.array(Y_pred.todense())
        print("       ==> Done with this case!\n")
        return Y_pred
    
def run_mass_for_one_test(X_train, Y_train, X_test, Y_test, X_unlabel=None):
    conf = mass_config[0]   
    classifier = Classifier(conf)
    classifier.fit(X_train, Y_train, X_unlabel)
    Y_pred = classifier.predict(X_test)
    score = Evaluation.micro_evaluation(Y_test, Y_pred)
    return score

def run_mass_for_every_test_of_lda(Features_T4, Labels_T4, Test_Features, Test_Labels, LDA_Features_T4, lda_case):

    report_columns = ['train', 'test', 'Precision', 'Recall', 'F1']
    report = pd.DataFrame(columns=report_columns)

    for i in range(len(Features_T4)):
        newFeatures = pd.concat([Features_T4[i], LDA_Features_T4[i].iloc[:T4_size[i], : ]], axis=1)
        newFeatures = newFeatures.fillna(0)
        new_Test_Features = pd.concat([Test_Features, LDA_Features_T4[i].iloc[T4_size[i] : , : ]], axis=1)
        new_Test_Features = new_Test_Features.fillna(0)
        print("Train contain NAN:", newFeatures.isnull().any().any())
        print("Test contain NAN:", new_Test_Features.isnull().any().any())

        X_train = np.array(newFeatures); 
        X_test = np.array(new_Test_Features); 
        
        Y_train = np.array(Labels_T4[i])
        Y_test = np.array(Test_Labels)

        
        config_description ="%s - %d train, %d test" % (lda_case, X_train.shape[0], X_test.shape[0])
        print(config_description)
        score = run_mass_for_one_test(X_train, Y_train, X_test, Y_test, X_train[0:50, : ])
        report.loc[i] = [X_train.shape[0], X_test.shape[0], score[0], score[1], score[2]]

    return report
        

def run_mass_for_all_lda_cases():

    Features_T4 = []
    Labels_T4 = []

    for i in range(len(T4_size)):
        features = pd.read_csv("data/D4i/D4" + T4_name[i] + "_features.txt", header=None, sep=",", dtype=np.float32)
        labels = pd.read_csv("data/D4i/D4" + T4_name[i] + "_labels.txt", header=None, sep=",", dtype= np.float32)
        Features_T4.append(features)
        Labels_T4.append(labels)

    Test_Features = pd.read_csv("data/D4i/test_features.txt", header=None, sep=",", dtype=np.float32)
    Test_Labels = pd.read_csv("data/D4i/test_labels.txt", header=None, sep=",", dtype= np.float32)

    #LDA Here mean AMC
    #Change here
    writer = pd.ExcelWriter("./output/AMC-10topics-result.xlsx")
    
    #Change here
    lda_names = ["AMC_10"]#, "LDA_10", "LDA_15", "LDA_25", "LDA_50"]
    # Read LDA_Features (Train, Test) From LDA_Res/LDA_5/D4a_text_data/dtopicdist
    
    for lda_case in lda_names:
        LDA_Features_T4 = []
        for i in range(len(T4_size)):
            #Change here
            path = "data/Exp2_Res/Output10/D4" + T4_name[i] + "_text_data/D4" + T4_name[i] + "_text_data.dtopicdist"
            print(path)
            lda_features = pd.read_csv(path, header=None, sep=" ", dtype=np.float32)
            LDA_Features_T4.append(lda_features)

        report = run_mass_for_every_test_of_lda(Features_T4, Labels_T4, Test_Features, Test_Labels, LDA_Features_T4, lda_case)

        print(report)
        report.to_excel(writer, lda_case)
    

    writer.save()
    
    return

if __name__ == "__main__":
    print("Run")
    run_mass_for_all_lda_cases()
