# kmer count
import keras
import json, math,sys
import sys,random
import numpy as np
import datetime, os, re
from sklearn import metrics
from sklearn.svm import LinearSVC as SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR

K = 6

def hashGenerate(sub_seq):  #input sequence with length = K
    dic = {'a':0,'A':0,'t':1,'T':1,'c':2,'C':2,'g':3,'G':3}
    hash_index = 0
    if len(sub_seq)!= K:
        print('Length Error')
        sys.exit()
    for i in range(K):
        hash_index = hash_index + dic[sub_seq[i]]*(4**(K-i-1))
    return hash_index

def NNtest(sub_seq):
    if len(sub_seq)!= K:
        print('Length Error')
        sys.exit()
    for i in range(K):
        if sub_seq[i]=='N':
            return False
    else:
        return True

def slidingWindow(sequence):#k-mer vector
    feature = np.zeros(4**K)
    L = len(sequence)
    for i in range(L-K+1):
        sub_seq = sequence[i:K+i]
        if NNtest(sub_seq)==True:
            feature[hashGenerate(sub_seq)]+=1
    return feature

def countKmer(filename):
    print cell,filename
    fin  = open(cell+'/'+filename)
    line = fin.readline()
    feats = []
    label = []
    for line in fin:
        line = line.strip().split(',')
        feature = slidingWindow(line[5])
        feats.append(feature)
        label.append(line[-2])
    fin.close()
    feats = np.array(feats)
    label = np.array(label)
    np.savez(cell+'/kmer/'+filename.split('.')[0]+'_kmer.npz', kmer=feats, label=label)

def RandomForest():
	# Random Forest classification
	clf=RF(n_estimators=100)
	clf.fit(X_train,Y_train)
	y_pred = clf.predict(X_val)
	y_score = clf.predict_proba(X_val)
	y_score = y_score[:,1]
	auroc = metrics.roc_auc_score(Y_val, y_score)

def logistic():
	# lr classification
	clf=LR()
	clf.fit(X_train,Y_train)
	y_pred = clf.predict(X_val)
	y_score = clf.predict_proba(X_val)
	y_score = y_score[:,1]
	auroc = metrics.roc_auc_score(Y_val, y_score)

def SVM():
	# SVM classification
	clf=SVC()
	clf.fit(X_train,Y_train)
	y_pred = clf.predict(X_val)
	y_score = clf.decision_function(X_val)
	auroc = metrics.roc_auc_score(Y_val, y_score)

def load_data():
	X_train_1 = np.load(cell+'/kmer/enhancer1_B_kmer.npz')['kmer'].astype('float32')
	X_train_2 = np.load(cell+'/kmer/enhancer2_B_kmer.npz')['kmer'].astype('float32')
	X_train = np.hstack((X_train_1, X_train_2))
	Y_train = np.load(cell+'/kmer/enhancer1_B_kmer.npz')['label'].astype('int32')
	X_train_1 = 0
	X_train_2 = 0
	print X_train.shape,Y_train.shape

	X_val_1 = np.load(cell+'/kmer/enhancer1_C_kmer.npz')['kmer'].astype('float32')
	X_val_2 = np.load(cell+'/kmer/enhancer2_C_kmer.npz')['kmer'].astype('float32')
	X_val = np.hstack((X_val_1, X_val_2))
	Y_val = np.load(cell+'/kmer/enhancer1_C_kmer.npz')['label'].astype('int32')
	return X_train,Y_train,X_val,Y_val

