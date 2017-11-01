import numpy as np 
import pandas as pd 
from scipy import sparse

from sklearn.model_selection import KFold
from utils.config import *
from eval import Evaluation

import matplotlib.pyplot as plt

class Classifier(object):
    
    def __init__(self, conf):
        print(conf["name"] + " cclassifier init!")
        self.conf = conf
        return

    def fit(self, X_train, Y_train, X_unlabel=None):
        print(self.conf["name"] + " fit!")
        if self.conf["name"] != "MASS":
            self.classifier = self.conf["classifier"]
            Y_train = sparse.csr_matrix(Y_train)
            self.classifier.fit(X_train, Y_train)

        else:
            X_train = sparse.csr_matrix(X_train)
            X_unlabel = sparse.csr_matrix(X_unlabel)
            
            self.classifier = self.conf["classifier"](n_features=X_train.get_shape()[1],n_labels=Y_train.shape[1])
            self.classifier.fit(X_train, Y_train, X_unlabel)
        return self
    
    def predict(self, X_test):
        print(self.conf["name"] + " predict!")
        X_test = sparse.csr_matrix(X_test)           
        Y_pred = self.classifier.predict(X_test)
        if self.conf["name"] != "MASS":
            Y_pred = np.array(Y_pred.todense())
        return Y_pred

def run_kfold(classifier, eval, X, Y, X_unlabel=None, kfolds=5):
    #Perform K-Fold
    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    precision, recall, f_score = 0, 0, 0
    
    for train_index, test_index in kf.split(X):
        X_train, Y_train = X[train_index], Y[train_index]
        X_test, Y_test = X[test_index], Y[test_index]
        X_test = sparse.csr_matrix(X_test)
        X_train = sparse.csr_matrix(X_train)
        if X_unlabel is None:
            classifier.fit(X_train, Y_train)
        else:
            classifier.fit(X_train, Y_train, X_unlabel)
        Y_pred = classifier.predict(X_test)
        precision_result = eval(Y_test, Y_pred)[0]
        recall_result = eval(Y_test, Y_pred)[1]
        f_result = eval(Y_test, Y_pred)[2]
        precision += precision_result
        recall += recall_result
        f_score += f_result

    return (precision/kfolds, recall/kfolds, f_score/kfolds)
    
def run_classification(classifier_config, mass_config, X, Y, X_unlabel):
    print("run classification!")

    f_score = {}
    precision = {}
    recall = {}

    n_samples = [500, 600, 800, 1000, 1200]
    for n_sample in n_samples:
        for conf in classifier_config:    
            classifier = Classifier(conf)
            score = run_kfold(classifier, Evaluation.micro_evaluation, X[:n_sample, :], Y[:n_sample, :])
            if f_score.get(conf["name"]) is None:
                f_score[conf["name"]] = list()
                precision[conf["name"]] = list()
                recall[conf["name"]] = list()
            precision[conf["name"]].append(score[0])
            recall[conf["name"]].append(score[1])
            f_score[conf["name"]].append(score[2])
        
        for conf in mass_config:
            classifier = Classifier(conf)
            score = run_kfold(classifier, Evaluation.micro_evaluaton, X[:n_sample, :], Y[:n_sample, :], X_unlabel)
            if f_score.get(conf["name"]) is None:
                f_score[conf["name"]] = list()
                precision[conf["name"]] = list()
                recall[conf["name"]] = list()
            precision[conf["name"]].append(score[0])
            recall[conf["name"]].append(score[1])
            f_score[conf["name"]].append(score[2])
    
    return (precision, recall, f_score)

def run_test(X, Y, X2):
    print("Execute!")
    micro_score = run_classification(classifier_configuration, mass_configuration, X, Y, X2)
    print(micro_score)
    n_samples = [500, 600, 800, 1000, 1200]
    export_score(micro_score, n_samples)
    plot_score(micro_score[2], n_samples)
    return

def export_score(micro_score, n_samples):
    print("plot-score!")
    text_file = open("Test_eval.txt", "w")
    text_file.write("BN_Features with the number of Unlabeled Data = 100")
    text_file.write("n_samples: ")
    text_file.write("\n")
    text_file.write(str(n_samples))
    text_file.write("\n")
    for key in micro_score[0]:
        text_file.write(key)
        text_file.write("\n")
        text_file.write("precision")
        text_file.write("\n")
        text_file.write(str(micro_score[0][key]))
        text_file.write("\n")
        text_file.write("recall")
        text_file.write("\n")        
        text_file.write(str(micro_score[1][key]))
        text_file.write("\n")     
        text_file.write("f_score")
        text_file.write("\n")        
        text_file.write(str(micro_score[2][key]))
        text_file.write("\n")   
        text_file.write("\n")   
    text_file.close()
    return

def plot_score(f1score, n_samples): 
    import matplotlib.pyplot as plt
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    for key in f1score:
        print(key)
        p = ax.plot(n_samples, f1score[key], 'o-', label = key)
    lgd = ax.legend( loc='center right', bbox_to_anchor=(1.5, 0.5))
    
    ax.set_ylabel('F1-Score')
    ax.set_xlabel('The number of samples')
    plt.ylim((0, 1))
    fig.savefig('image_output.png', dpi=300, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    return 

#########################################################################################    
if __name__ == "__main__":
    print("Experiment! Hello")
    features = np.loadtxt("data/BN_Features.txt")
    labels = pd.read_csv("data/label_TFIDF-LDA_10topics.csv", header=None)
    X = np.array(features[0:1200,:])
    Y = np.array(labels.iloc[0:1200,:])
    X2 = np.array(features[1300:1400,:])
    run_test(X, Y, X2)
   