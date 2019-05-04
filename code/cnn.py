#!/usr/bin/env python
#keras version: keras-1.2.0

import os
import sys
import keras
import datetime
import numpy as np
import hickle as hkl
from sklearn import metrics
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers import LSTM, Bidirectional
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Reshape, Merge, Permute
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
from keras.models import load_model
from keras.engine.topology import Layer, InputSpec
from keras import initializations

RESIZED_ENHANCER_LEN = 2000
RESIZED_PROMOTER_LEN = 2000
SEQ_LEN = 2000


# Attention GRU network
class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializations.get('normal')
        #self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        #self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        #self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        M = K.tanh(x)
        alpha = K.dot(M,self.W)#.dimshuffle(0,2,1)

        ai = K.exp(alpha)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
        weighted_input = x*weights.dimshuffle(0,1,'x')
        return K.tanh(weighted_input.sum(axis=1))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

def model_def():
    drop_rate = 0.5 
    conv_enhancer_seq = Sequential()
    conv_enhancer_seq.add(Convolution2D(1024, 4, 40, activation = 'relu', border_mode = 'valid',
                                        dim_ordering = 'th', input_shape = (1, 4, RESIZED_ENHANCER_LEN)))
    conv_enhancer_seq.add(MaxPooling2D(pool_size = (1, 20), border_mode = 'valid', dim_ordering = 'th'))
    conv_enhancer_seq.add(Reshape((1024, 98)))
    conv_promoter_seq = Sequential()
    conv_promoter_seq.add(Convolution2D(1024, 4, 40, activation = 'relu', border_mode = 'valid',
                                        dim_ordering = 'th', input_shape = (1, 4, RESIZED_PROMOTER_LEN)))
    conv_promoter_seq.add(MaxPooling2D(pool_size = (1, 20), border_mode = 'valid', dim_ordering = 'th'))
    conv_promoter_seq.add(Reshape((1024, 98)))
    merged = Sequential()
    merged.add(Merge([conv_enhancer_seq, conv_promoter_seq], mode = 'concat'))
    merged.add(Permute((2, 1)))
    merged.add(BatchNormalization())
    merged.add(Dropout(drop_rate))
    merged.add(Bidirectional(LSTM(100, return_sequences = True), merge_mode = 'concat'))
    merged.add(AttLayer())
    merged.add(BatchNormalization())
    merged.add(Dropout(drop_rate))
    model = Sequential()
    model.add(merged)
    model.add(Dense(925))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(1, activation = 'sigmoid'))
    return model

def evaluate_performance(y_truth, y_pred, p_y_given_x):
    TP = float(sum((y_truth == 1) & (y_pred == 1)))
    FP = float(sum((y_truth == 0) & (y_pred == 1)))
    FN = float(sum((y_truth == 1) & (y_pred == 0)))
    TN = float(sum((y_truth == 0) & (y_pred == 0)))
    ACC = (TP + TN) / (TP + FP + FN + TN)
    BER = 0.5 * (FN / (FN + TP + 1e-7) + FP / (FP + TN + 1e-7))
    P = TP / (TP + FP + 1e-7)
    R = TP / (TP + FN + 1e-7)
    F1 = 2 * P * R / (P + R + 1e-7)
    try:
        MCC = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    except:
        MCC = 0.0
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.metrics import roc_curve, precision_recall_curve
    auROC = roc_auc_score(y_truth, p_y_given_x)
    auPR = average_precision_score(y_truth, p_y_given_x)
    fpr, tpr, _ = roc_curve(y_truth, p_y_given_x)
    precision, recall, _ = precision_recall_curve(y_truth, p_y_given_x)
    return (ACC, BER, MCC, auROC, auPR, fpr, tpr, precision, recall, P, R, F1)

def evaluate(Y_val, y_pred, y_score):
    acc   = metrics.accuracy_score(Y_val, y_pred)
    print('Test acc: ', acc)
    precision = metrics.precision_score(Y_val, y_pred)
    print('Test precision: ', precision)
    recall = metrics.recall_score(Y_val, y_pred)
    print('Test recall: ', recall)
    f1    = metrics.f1_score(Y_val, y_pred)
    print('Test f1: ', f1) 
    mcc   = metrics.matthews_corrcoef(Y_val, y_pred)
    print('Test mcc: ', mcc)
    fpr, tpr, thresholds = metrics.roc_curve(Y_val, y_score, pos_label=1)
    fpr = np.asarray(fpr, dtype="str")
    tpr = np.asarray(tpr, dtype="str")
    thresholds = np.asarray(thresholds, dtype="str")
    auroc = metrics.roc_auc_score(Y_val, y_score)
    print('Test auroc: ', auroc)
    auprc = metrics.average_precision_score(Y_val, y_score)
    print('Test auprc: ', auprc)

    fout=open("./models/"+timestamp+"/hist.txt","a")
    fout.write('Test accuracy: '+str(acc)+'\n')
    fout.write('Test f1 score: '+str(f1)+'\n')
    fout.write('Test precision: '+str(precision)+'\n')
    fout.write('Test recall: '+str(recall)+'\n')
    fout.write('Test mcc: '+str(mcc)+'\n')
    fout.write('Test auroc: '+str(auroc)+'\n')
    fout.write('Test auprc: '+str(auprc)+'\n')
    fout.write('fpr'+'\t'+ '\t'.join(fpr.tolist())+'\n')
    fout.write('tpr'+'\t'+ '\t'.join(tpr.tolist())+'\n')
    fout.write('thresholds'+'\t'+ '\t'.join(thresholds.tolist())+'\n')
    fout.close()

def f1(y_true, y_pred):
    TP = K.sum(K.equal(y_true, 1) & K.equal(K.round(y_pred), 1))
    FP = K.sum(K.equal(y_true, 0) & K.equal(K.round(y_pred), 1))
    FN = K.sum(K.equal(y_true, 1) & K.equal(K.round(y_pred), 0))
    TN = K.sum(K.equal(y_true, 0) & K.equal(K.round(y_pred), 0))
    print TP,FP,FN,TN
    P = TP / (TP + FP + K.epsilon())
    R = TP / (TP + FN + K.epsilon())
    F1 = 2 * P * R / (P + R + K.epsilon())
    return F1

def train_and_evaluate(model):
    print('Loading data...')
    enhancer_shape = (-1, 1, RESIZED_ENHANCER_LEN, 4)
    promoter_shape = (-1, 1, RESIZED_PROMOTER_LEN, 4)
    enhancer_D = np.load(cell+'/enhancer1_B.npz')
    promoter_D = np.load(cell+'/enhancer2_B.npz')
    #
    n_D = enhancer_D['label'].shape[0]
    np.random.seed(n_D)
    rand_index_D = range(0, n_D)
    np.random.shuffle(rand_index_D)
    label_D = enhancer_D['label'][rand_index_D]
    enhancer_seq_D = enhancer_D['sequence'].astype('float32').reshape(enhancer_shape).transpose(0, 1, 3, 2)[rand_index_D]
    promoter_seq_D = promoter_D['sequence'].astype('float32').reshape(promoter_shape).transpose(0, 1, 3, 2)[rand_index_D]
    
    ## Train model
    model.compile(loss = 'binary_crossentropy',
                  optimizer = optimizers.Adam(lr = 0.00001),
                  metrics = ['acc', f1])
    filename = cell+'/models/best_model_seq_'+Type+'.h5'
    modelCheckpoint = ModelCheckpoint(filename, monitor = 'val_acc', save_best_only = True, mode = 'max')
    model.fit([enhancer_seq_D, promoter_seq_D], label_D, nb_epoch = 50, batch_size = 100,
              validation_split = 0.1, callbacks = [modelCheckpoint])

""" MAIN """
model = model_def()
os.system('mkdir -p '+cell+'/models-'+DEF)
train_and_evaluate(model)


