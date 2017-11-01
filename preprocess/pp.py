import numpy as np 
import pandas as pd
import os
import math
import subprocess
import codecs 

class Preprocess(object): 
    
    TYPE_NAME = ["BN", "TF-IDF"]

    def __init__(self, feature_type="BN", wordsg=False): 
        self.feature_type = feature_type
        self.wordsg = wordsg
        return

    def fit_transform(self, X):
        print("Preprocess - Fit_Transform") 
        self.fit(X)
        self.call_fit = True
        return self.transform(X)

    def fit(self, X): 
        print("Preproccess- Fit")
        self.build_dictionary(X)
        return
    
    def transform(self, X, fileName="C:/Users/Quang/Desktop/Python_MASS/src/models/dict.txt"): 
        print("Preprocess - transform")

        # Load dictionary
        if self.dict is None: 
            self.dict = self.readDict("C:/Users/Quang/Desktop/Python_MASS/src/models/dict.txt") 

        if self.feature_type == "BN": 
            return self.bn_vectorizer(X)

        elif self.feature_type == "TF-IDF": 
            return self.tfidf_vectorizer(X)

    def word_segmenter(self, X):
        print("Preprocess - Word Segmenter!")
        path = 'C:/Users/Quang/Desktop/Python_MASS/src/preprocess/wordSegmenter/'
        filename = "input.txt"
        file = codecs.open(path+filename, "w", "utf-8")
        for line in X: 
            file.write(line)
        file.close()
        print("done")
        
        outputname = "output.txt"
        cmdline = path + 'vnTokenizer.bat ' + ' -i ' + filename + ' -o ' + outputname
        subprocess.Popen(cmdline).wait()

        lines = []
        with open(path + outputname, encoding="utf8") as f: 
            lines = f.readlines()
        return lines
    
    def build_dictionary(self, X):
        print("Preprocess - Build Dictionary")
        self.dict = {} 
        if (self.wordsg == True):
            X = self.word_segmenter(X)
        feature = 0
        for line in X: 
            split_words = line.split()
            for word in split_words:
                word = word.lower() 
                if self.dict.get(word) is None:
                    self.dict[word] = feature
                    feature += 1
        self.writeDict(self.dict, "C:/Users/Quang/Desktop/Python_MASS/src/models/dict.txt")
        return self


    def bn_vectorizer(self, X):
        print("binary vector")
        matrix = np.zeros((len(X), len(self.dict)), dtype=float)
        for i in range(len(X)):
            split_words = X[i].split()
            for word in split_words:
                word = word.lower()
                j = self.dict.get(word)          
                matrix[i, j] = 1
        return matrix

    def tfidf_vectorizer(self, X):
        matrix = np.zeros((len(X), len(self.dict)), dtype=float)
        for i in range(len(X)):
            split_words = X[i].split()
            for word in split_words:
                word = word.lower()
                tf = self.compute_tf(word, X[i])
                idf = self.compute_idf(word, X)
                j = int(self.dict.get(word))  
                matrix[i, j] = tf * idf
        return matrix

    def compute_tf(self, word, sentence): 
        split_words = sentence.split()
        word_freq = split_words.count(word)
        tf = word_freq / len(split_words)
        return tf

    def compute_idf(self, word, X):
        n_docs_with_word = 0
        for i in range(len(X)): 
            if word in X[i]: 
                n_docs_with_word += 1
        
        n_docs = len(X)
        idf = n_docs / n_docs_with_word
        return math.log(idf)

    def writeDict(self, dict, filename, sep=" "):
        text_file = open(filename, "w", encoding="utf8")
        for word in dict:
            line = word + sep + str(dict[word])
            text_file.write(line)
            text_file.write("\n")
        text_file.close()

    def readDict(self, filename="C:/Users/Quang/Desktop/Python_MASS/src/models/dict.txt", sep=" "):
        dict = {}
        with open(filename, encoding="utf8") as f: 
            lines = f.readlines()
            for line in lines:
                elem = line.split(sep)
                dict[elem[0]] = int(elem[1])
        return dict
    
if __name__ == "__main__": 
    prc = Preprocess(feature_type="BN")
    lines = []
    
    with open("C:/Users/Quang/Desktop/Python_MASS/src/data/unlabeltext.txt", encoding="utf8") as f: 
        lines = f.readlines()
    df = pd.DataFrame(a)
    df.to_csv("C:/Users/Quang/Desktop/Python_MASS/src/data/BN_Features.txt", sep=" ", index=False, header=False)
    