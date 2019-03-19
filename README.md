# MASS Implementation
The implementatin for paper "A Semi-supervised Multi-label Classification Algorithm with Specific Features".


## Train and make prediction from MassClassifier in 30 seconds
The `MassClassifier` has several methods which are similar to general models in `sklearn`.

**30 seconds of code** 
```python
from mass import MassClassifier

# 1. Init a model
classifier = MassClassifier(n_labels = 5, n_features=2300)

# 2. Train the model
classifier.fit(X_train, Y_train, X_unlabel)

# 3. Make prediction
classifier.predict(X_test)
```

## A.1 How to call MassClassifier from command line
The following command is used to run the toolkit, the system needs python version
3.0 or newer:

```bash
python massclassifier.py <resources_dir> <input_file_dir> <output_file_dir> <options>
```

The arguments can be described as follows:
* `<resources_dir>` is the directory containing a learned model (The default model is
for multi-label classification for Vietnamese hotel).
* `<input_file_dir>` is the directory containing the input file. In the input file, each
line represents for the list of features of an instance.
* `<output_file_dir>` is the directory containing the output file. Each line contains a
label indicator of binary value {0, 1}.
* `<options>` the type of label space division technique “random” for random
approach, and “dd” for data driven approach.

## A.2 How to use APIs

###1. Initialize a classifier

Firstly, we have to create a classifier:
```python
mClassifier = MassClassifier(learned_model_path, lspace_type="random",
                             required_dense=True)
```

* `learned_model_path`: if we want to use our trained model, we have to pass a path
of the learned model for the argument “learned_model_path. Otherwise, we will skip
this argument and the default value for this: learned_model_path = None.
* `lspace_type`: This argument defines the strategy this classifier will adopt to train a
model. “random” for random approach, and “dd” for data driven approach.
* `required_dense`: The representation of features would be dense matrix (True) or
sparse matrix (False).

### 2. Train a new model
After initializing a classifier, we can use the following method to train a new model from
our data:

```python
mClassifier.fit(X_train, Y_train, X_unlabel)
```

* `X_train`: A matrix of features representing for a set of labeled instances <`n1_samples` x `n_features`>
* `Y_train`: A matrix of labels corresponding to each of `X_train` of <`n1_samples` x `n_labels`>
* `X_unlabel`: A matrix of features representing for a set of unlabeled instances.
<`n2_samples` x `n_features`>

>> `X_train`, `Y_train`, `X_unlabel` can be represented by dense matrices or sparse matrices.
The classifier will ultimately transform dense matrices to sparse matrices.

Then the learned model will be exported into a file at the directory declared at the former
step.

### 3. Predict a set of labels for unseen instances
```python
Y_pred = mClassifier.predict(X_test)
```

* `X_test`: A feature matrix of instances we need to predicts their labels.
* This method will return a predicted matrix of labels for `X_test`, here we store it to
the variable `Y_pred`.

### 4. Evaluate the performance of a trained model. 
```python
result = Evaluation.eval(Y_test, Y_pred)
print(result)

```
* `Y_test` here is a label matrix of instances for testing or their truly observed labels.
* `Y_pred`, as predicted in step 3, are a matrix of predicted labels for testing instances.

The result should display the value as `Precision` = `a%`, `Recall` = `b%` and `F1-score` = `c%`.


## B. Build the wrapper Classifier
The operations on sparse matrices lead to better speed. Please edit the file `config.py` to build a wrapper.

```python
from scipy import sparse
import numpy as np

class Classifier(object):
    
    def __init__(self, conf):
        print("       0. Init " + conf["name"] + " classifier\n")
        self.conf = conf
        

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

```

## C. Concreate Examples
```python
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
```    