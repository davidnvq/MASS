import numpy as np 
import pandas as pd
import os
import math
import subprocess
import codecs 

class Preprocess(object): 
    
    TYPE_NAME = ["BN", "TF-IDF"]

    def __init__(self, feature_type="BN"): 
        self.feature_type = feature_type
        self.filename = ""
        self.data_wd = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")), "data")
        self.binary_data_wd = os.path.join(self.data_wd, "LDA_T4")
        self.text_data_wd = os.path.join(self.data_wd, "text_data/D4i_text_data")
        print(self.text_data_wd)
        return

    def fit(self, filename):
        self.filename = filename + ".txt"
        self.text_path = os.path.join(self.text_data_wd, self.filename)
        self.dict_path = os.path.join(self.binary_data_wd, filename + ".vocab")
        self.bin_text_path = os.path.join(self.binary_data_wd, filename + ".docs")

        print("Preproccess- Fit")
        with open(self.text_path, encoding="utf8") as f:
            self.X = f.readlines()

        self.build_dictionary()
        return
    
    def transform(self): 
        print("Preprocess - transform")

        # Load dictionary
        if self.dict is None: 
            self.dict = self.readDict() 

        if self.feature_type == "BN": 
            newlines = self.bn_vectorizer(self.X)   
            

        elif self.feature_type == "TF-IDF": 
            return self.tfidf_vectorizer(self.X)

        text_file = open(self.bin_text_path, "w", encoding="utf8")
        for line in newlines: 
            text_file.write(line)
            text_file.write("\n")
        text_file.close()
        return newlines

    def build_dictionary(self):
        print("Preprocess - Build Dictionary")
        self.dict = {} 
        
        feature = 0
        for line in self.X: 
            split_words = line.split()
            for word in split_words:
                word = word.lower() 
                if self.dict.get(word) is None:
                    self.dict[word] = feature
                    feature += 1
        self.writeDict(self.dict, self.dict_path)
        return self

    def bn_vectorizer(self, X):
        lines = []

        print("binary vector")
        matrix = np.zeros((len(X), len(self.dict)), dtype=float)
        for i in range(len(X)):
            line = ""
            split_words = X[i].split()
            for word in split_words:
                word = word.lower()
                j = self.dict.get(word)
                line += str(j) + " "
            line = line[0:len(line)-1]
            lines.append(line)
        return lines

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

    def writeDict(self, dict, filename, sep=":"):
        text_file = open(filename, "w", encoding="utf8")
        for word in dict:
            line = str(dict[word]) + sep + word
            text_file.write(line)
            text_file.write("\n")
        text_file.close()

    def readDict(self, filename, sep=":"):
        dict = {}
        with open(filename, encoding="utf8") as f: 
            lines = f.readlines()
            for line in lines:
                elem = line.split(sep)
                dict[elem[1]] = int(elem[0])
        return dict


    
if __name__ == "__main__":
    # Chuyen file text sang dang .vocab .docs 
    prc = Preprocess(feature_type="BN")
    #prc.fit("D4a_text_data")
    #prc.fit("D4b_text_data")
    #prc.fit("D4c_text_data")
    #prc.fit("D4d_text_data")
    #prc.fit("D4e_text_data")
    prc.fit("D4f_text_data")
    prc.transform()

