import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
 
from scipy import sparse
from sklearn.model_selection import KFold
from config import *
from eval import Evaluation

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
        print("       2. Predict Y_pred for X_test from ")
        X_test = sparse.csr_matrix(X_test)           
        Y_pred = self.classifier.predict(X_test)
        if self.conf["name"] != "MASS":
            Y_pred = np.array(Y_pred.todense())
        print("       ==> Done with this fold!\n")
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
    
def run_classification(classifier_config, mass_config, X, Y, X_unlabel, n_labeled_samples):
    f_score = {}
    precision = {}
    recall = {}

    for n_sample in n_labeled_samples:
        for conf in classifier_config:
            print("\n\nTraining 5-fold %s with (%d labeled samples) and (%d unlabeled samples)" % (conf["name"], n_sample, 0))
            classifier = Classifier(conf)
            score = run_kfold(classifier, Evaluation.micro_evaluation, X[:n_sample, :], Y[:n_sample, :], kfolds=5)
            if f_score.get(conf["name"]) is None:
                f_score[conf["name"]] = list()
                precision[conf["name"]] = list()
                recall[conf["name"]] = list()
            precision[conf["name"]].append(score[0])
            recall[conf["name"]].append(score[1])
            f_score[conf["name"]].append(score[2])
        
        for conf in mass_config:
            print("\n\nTraining 5-fold %s with (%d labeled samples) and (%d unlabeled samples)" % (conf["name"], n_sample, X_unlabel.shape[0]))
            classifier = Classifier(conf)
            score = run_kfold(classifier, Evaluation.micro_evaluation, X[:n_sample, :], Y[:n_sample, :], X_unlabel, kfolds=5)
            if f_score.get(conf["name"]) is None:
                f_score[conf["name"]] = list()
                precision[conf["name"]] = list()
                recall[conf["name"]] = list()
            precision[conf["name"]].append(score[0])
            recall[conf["name"]].append(score[1])
            f_score[conf["name"]].append(score[2])

    return (precision, recall, f_score)

def export_score(micro_score, n_samples):
    text_file = open("./output/Test_eval.txt", "w")
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
    fig.savefig('./output/image_output.png', dpi=300, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    return 


"""
Đây là file chạy baseline của khoá luận tổng quát chạy cho rất nhiều các mô hình phân lớp khác nhau để so sánh baseline với MASS.
Tuy nhiên, nếu c chỉ cần chạy MASS không thôi thì trong file tham số config.py set mỗi mass_config thôi, còn cái classifier_configuration set rỗng.
"""
def run_test(X, Y, X2, n_labeled_samples):
    # đặt tạm cái classifier_config = rỗng, vì mình chạy mỗi mass mà
    classifier_config = []
    micro_score = run_classification(classifier_config, mass_config, X, Y, X2, n_labeled_samples)
    print(micro_score)

    export_score(micro_score, n_labeled_samples)
    plot_score(micro_score[2], n_labeled_samples)
    return

if __name__ == "__main__":
    """
    If we use BN_Features.txt -> BN-outer-Test-eval.txt
    If we use feature_TFIDF-LDA-10topics.csv -> TFIDF-LDA-10topics-outer-Test-eval
    """

    """
    Dữ liệu có nhãn là dữ liệu gồm có features và labels tương ứng, có nghĩa là một cặp hoàn chỉnh (features, labels)
    Mỗi nhận xét được biểu diễn bởi một vector features, và 1 vector labels. Hồng xem cái file dữ liệu để hiểu thêm nhé.  
    Để thu được dữ liệu không nhãn, ta chỉ cần bỏ labels tức là một nhận xét chỉ có (features, )
    """

    """
    Features ở đây là một ma trận, hiểu đơn giản, mỗi hàng trong này tương ứng với một vector features của nhận xét. 
    Labels cũng tương tự.
    """
    """
    Ở đây, file features_TFIDF.csv là file dữ liệu thuần gốc, biểu diễn mỗi câu dưới dạng TFIDF
    Trong trường hợp của nhóm, mình muốn làm giàu thêm đặc trưng bằng cách thêm đặc trưng xác suất chủ đề của câu
    thì đơn giản là nối đặc trưng LDA (LTM) vào đuôi của dữ liệu thuần gốc. File này chính là file model-final.tassign trong LDA ấy. 
    C phải chú ý là số lượng vector ở 2 file này là như nhau (tức có số dòng giống nhau)
    """
    Features = pd.read_csv("data/features_TFIDF.csv", header=None, sep=",", dtype=np.float32) 
    Labels = pd.read_csv("data/labels.csv", header=None, sep=",", dtype= np.float32)
    print(Features.shape)
    # Ở đây là ví dụ thôi, chứ file model-final.tassign này không chạy được nhé. Do hôm qua 2 chị em chạy nhầm.
    # Bỏ dấu # để chạy nhé.
    LDA_Features = pd.read_csv("data/model-final.theta", header=None, sep= " ", dtype=np.float32)
    
    # Ghép 2 file này lại với nhau. Bỏ dấu comment đi là chạy được nếu file model-final.tassign đúng format
    Features = pd.concat([Features, LDA_Features.iloc[:, : LDA_Features.shape[1]-1]], axis=1)
    Features = Features.fillna(0)
    
    """
    Chia dữ liệu thành tập có nhãn (X1, Y1)
    và tập không nhãn (X2, )
    """
    
    """
    Xác định xem số lượng dữ liệu có nhãn đưa vào huấn luyện là bao nhiêu? 
    Ví dụ: trong 1493 nhận xét, thì lấy 1300 nhận xét là dữ liệu có nhãn, số còn lại là dữ liệu không nhãn. (Thường thì tỉ lệ không nhãn/có nhãn ~ 1/10 -> 2/10 cho kết quả tốt nhất)
    """
    n_X1 = 1200

    X1 = np.array(Features.iloc[0:n_X1, : ])  # Ví dụ: Lấy các câu từ 1 -> 1200 làm tập có nhãn
    Y1 = np.array(Labels.iloc[0:n_X1, : ])    # Tương tự cho Y1,
    X2 = np.array(Features.iloc[n_X1: , :]) # các câu từ 1200 -> 1300 làm tập không nhãn   
    
    """
    Trong dữ liệu có nhãn X1, chúng ta muốn chạy thực nghiệm với các trường hợp có dữ liệu có nhãn là bao nhiêu?
    Liệt kê ra.
    Ví dụ tập dưới là chạy với dữ liệu có nhãn là 500, và 800
    Như vậy chúng ta sẽ thực hiện 5-fold trên 2 trường hợp sau: 
    1. Dữ liệu [có nhãn X1 = 500, không nhãn X2 = 1493 - n_X1 = 293]
    2. Dữ liệu [có nhãn X1 = 800, không nhãn X2 = 1493 - n_X1 = 293]
    """
    n_labeled_samples = [500, 800]#, 1000, 1200]
    
    """
    Cái này chạy như sau:
    - Đầu vào:
        Dữ liệu X đặc trưng (X này có thể là tf-idf hay là đặc trưng tf-idf đã gán thêm chủ đề xác suất (cái này gọi là làm giàu dặc trưng)
        và dữ liệu Y nhãn.
    

    Dữ liệu (Features, Labels) được chia làm 2 phần (X1, Y1) là dữ liệu có nhãn, (X2, ) là dữ liệu không nhãn.    
    
    Để lấy tỉ lệ số lượng dữ liệu có nhãn, ví dụ 500, 700, 1000 nhận xét có nhãn thì sửa cái phần n_samples nhé
    Với mỗi một lượng dữ liệu có nhãn, và không có nhãn -> huấn luyện thu được 1 classifier. 
    Dùng k-fold nên cái này trả luôn về P, R, F1. Không cần tính X_test, Y_test làm gì nhé.

    - Đầu ra: 
        Sẽ là kết quả ở thư mục output, file Test_eval.txt export ấy, gồm đầy đủ, precision, recall, rồi F1 của từng trường hợp, từng mô hinh phân lớp (MASS là 1 trong số đó)
        Có vẽ luôn plot  
    """
    run_test(X1, Y1, X2, n_labeled_samples)
    