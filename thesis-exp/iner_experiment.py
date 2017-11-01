import numpy as np 
import pandas as pd 
from scipy import sparse

from sklearn.model_selection import KFold
from utils.config import *
from eval import Evaluation
from sklearn.model_selection import train_test_split

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

def run_average(classifier, eval, X, Y, X_unlabel=None, kfolds=5):
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
    
def run_classification(mass_config, X, Y, X_unlabel):
    print("run classification!")

    n_dlabels = [500, 750, 1000, 1250]
    n_dunlabels = [0, 50, 100, 150, 200, 250]
    
    config = mass_config[0]
    overall_score = []

    for n_dunlabel in n_dunlabels:
        indx = np.random.randint(X.shape[0], size=n_dunlabel)
        X2 = X[indx, :]
        for n_dlabel in n_dlabels: 
            index = np.random.randint(X.shape[0], size=n_dlabel)
            X_t, Y_t = X[index, :], Y[index, :]
            classifier = Classifier(config)
            score = run_average(classifier, Evaluation.micro_evaluaton, X_t, Y_t, X2)
            overall_score.append((n_dlabel, n_dunlabel, score[0], score[1], score[2]))    
    return overall_score

def run_test(X, Y, X2):
    print("Execute!")
    micro_score = run_classification(mass_configuration, X, Y, X2)
    print(micro_score)
    n_samples = [500, 600, 800, 1000, 1200]
    export_score(micro_score)
    #plot_score(micro_score)
    return

def export_score(micro_score):
    print("plot-score!")
    text_file = open("inner_Test_eval.txt", "w")
    text_file.write("BN_Features")
    text_file.write("\n")
    text_file.write(str(n_samples))
    text_file.write("\n")
    for key in micro_score:
        text_file.write(str(key[0]))
        text_file.write("\t")
        text_file.write(str(key[1]))
        text_file.write("\t")
        text_file.write(str(key[2]))
        text_file.write("\t")
        text_file.write(str(key[3]))
        text_file.write("\t")
        text_file.write(str(key[4]))
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
    print("Experiment!")
    features = np.loadtxt("data/BN_Features.txt")
    labels = pd.read_csv("data/label_TFIDF-LDA_10topics.csv", header=None)
    X = np.array(features[0:1250,:])
    Y = np.array(labels.iloc[0:1250,:])
    X2 = np.array(features[1240:1493,:])
    run_test(X, Y, X2)
   