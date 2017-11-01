import numpy as np 
import pandas as pd

from scipy import sparse
from eval import Evaluation


mass_config = [{"name": "MASS", "classifier" : MassClassifier}]

classifier_config = [{"name":"CC - SVM", "classifier" : 'class CC_SVM vào chỗ này'}, 
               {"name":"BR - SVM", "classifier" : 'class_SVM_vao_cho_nay'}
]

unlabel_config = [0, 100, 200, 300]
label_train_config = [500, 750, 1000]
label_test_config = [250]





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
    
def run_mass_for_one_test(X_train, Y_train, X_test, Y_test, X_unlabel):
    conf = mass_config[0]   
    classifier = Classifier(conf)
    classifier.fit(X_train, Y_train, X_unlabel)
    Y_pred = classifier.predict(X_test)
    score = Evaluation.micro_evaluation(Y_test, Y_pred)
    return score

def indices_split(label_train_size, label_test_size, unlabel_size):
    indices = np.array([i for i in range(1500)])
    np.random.seed(23031994)

    # Get test_indices
    test_indices = np.random.choice(indices.shape[0], size = label_test_size, replace=False)
    other_indices = [i for i in range(indices.shape[0]) if i not in test_indices]

    test_indices = indices[test_indices]; test_indices.sort()
    other_indices = indices[other_indices]; other_indices.sort()

    # Get train_indices
    train_indices = np.random.choice(other_indices.shape[0], size=label_train_size, replace=False)
    rest_indices = [i for i in range(other_indices.shape[0]) if i not in train_indices]

    train_indices = other_indices[train_indices]; train_indices.sort()
    other_indices = other_indices[rest_indices]; other_indices.sort()

    # Get un_label_indices
    unlabel_indices = np.random.choice(other_indices.shape[0], size=unlabel_size, replace=False)
    unlabel_indices = other_indices[unlabel_indices]; unlabel_indices.sort()
    
    return train_indices, test_indices, unlabel_indices

def run_mass_for_every_test_of_lda_case(Features, Labels, lda_case):
    unlabel_config = [0, 100, 200, 250]
    label_train_config = [500, 750, 1000]
    label_test_size = 250
    report_columns = ['STT', 'train', 'unlabel', 'test', 'Precision', 'Recall', 'F1']
    report = pd.DataFrame(columns=report_columns)

    order = 0
    for label_train_size in label_train_config:
        for unlabel_size in unlabel_config:
            order += 1
            train_indices, test_indices, unlabel_indices = indices_split(label_train_size, label_test_size, unlabel_size)
            X_train, Y_train = np.array(Features.iloc[train_indices.tolist(), : ]), np.array(Labels.iloc[train_indices.tolist(), : ])
            X_test, Y_test = np.array(Features.iloc[test_indices.tolist(), : ]), np.array(Labels.iloc[test_indices.tolist(), : ])
            X_unlabel = np.array(Features.iloc[unlabel_indices.tolist(), : ])

            config_description ="%s - %d train, %d unlabel, %d test" % (lda_case, X_train.shape[0], X_unlabel.shape[0], X_test.shape[0])
            print(config_description)

            score = run_mass_for_one_test(X_train, Y_train, X_test, Y_test, X_unlabel)
            
            # For excel file
            report.loc[order] = [order, label_train_size, unlabel_size, label_test_size, score[0], score[1], score[2]]
    return report    

def run_mass_for_all_lda_cases(Features, Labels):

    writer = pd.ExcelWriter("./output/NoLDA-result.xlsx")
    lda_names = ["No-LDA"]#, "LDA10", "LDA15", "LDA25", "LDA50", "LDA100"]
    print("Features", Features.shape)
    print("Labels", Labels.shape)
    for lda_case in lda_names:
        newFeatures = Features.fillna(0)
        newFeatures = pd.concat([newFeatures, newFeatures.iloc[0:10, : ]], axis = 0)
        newLabels = pd.concat([Labels, Labels.iloc[0:10, : ]], axis = 0)
        report = run_mass_for_every_test_of_lda_case(newFeatures, newLabels, lda_case)
        print(report)
        report.to_excel(writer, lda_case)
    writer.save()

    lda_folder = []

    return

if __name__ == "__main__":
    Features = pd.read_csv("data/features_TFIDF.csv", header=None, sep=",", dtype=np.float32)
    print (Features.shape[0]) 
    Labels = pd.read_csv("data/labels.csv", header=None, sep=",", dtype= np.float32)
    run_mass_for_all_lda_cases(Features, Labels)
