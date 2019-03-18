from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import zero_one_loss
from sklearn.metrics import coverage_error
import numpy as np
import pandas as pd


def micro_evaluation(Y_true, Y_pred):
    precision, recall, f1_score = 0, 0, 0
    n_true_pred_labels, n_real_labels, n_pred_labels = 0, 0, 0

    for i in range(Y_true.shape[0]):
        labels = Y_true[i]
        pred_labels = Y_pred[i]

        n_real_labels += np.sum(labels)
        n_pred_labels += np.sum(pred_labels)
        
        for i, j in zip(labels, pred_labels):
            if i == 1 and j == 1:
                n_true_pred_labels += 1 

    recall = n_true_pred_labels/n_real_labels
    precision = n_true_pred_labels/n_pred_labels
    f1_score = 2 * (recall * precision) / (precision + recall)
    #return np.array([precision, recall, f1_score])	
    return precision, recall, f1_score

def hammingloss(Y_true, Y_pred):
    return hamming_loss(Y_true, Y_pred)

def zerooneloss(Y_true, Y_pred):
    return zero_one_loss(Y_true, Y_pred)

def macro_evaluation(Y_true, Y_pred):
    precision = []
    recall = []
    f1score = []

    for i in range(Y.shape[1]): 
        precision.append(precision_score(Y_true[:,i], Y_pred[:, i]))
        recall.append(recall_score(Y_true[:,i], Y_pred[:, i]))
        f1score.append(f1_score(Y_true[:,i], Y_pred[:, i]))
    return (precision, recall, f1score)


def micro_evaluaton(Y_true, Y_pred):
    
    precision, recall, f1_score = 0, 0, 0
    n_true_pred_labels, n_real_labels, n_pred_labels = 0, 0, 0
    #text_file = open("Test_eval.txt", "w")
    for i in range(Y_true.shape[0]):
        labels = Y_true[i]
        pred_labels = Y_pred[i]

        n_real_labels += np.sum(labels)
        n_pred_labels += np.sum(pred_labels)
        
        for i, j in zip(labels, pred_labels):
            if i == 1 and j == 1:
                n_true_pred_labels += 1 
        
        #text_file.write(str(labels))
        #text_file.write("\n")
        #text_file.write(str(pred_labels))
        #text_file.write("\n\n") 
    n_real_labels -= 10
    #text_file.close()

    recall = n_true_pred_labels/n_real_labels
    precision = n_true_pred_labels/n_pred_labels
    f1_score = 2 * (recall * precision) / (precision + recall)
    return np.array([precision, recall, f1_score])				
    return precision, recall, f1_score
