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
    conv_enhancer1_seq = Sequential()
    conv_enhancer1_seq.add(Convolution2D(1024, 4, 40, activation = 'relu', border_mode = 'valid',
                                        dim_ordering = 'th', input_shape = (1, 4, SEQ_LEN)))
    conv_enhancer1_seq.add(MaxPooling2D(pool_size = (1, 20), border_mode = 'valid', dim_ordering = 'th'))
    conv_enhancer1_seq.add(Reshape((1024, 98)))
    conv_enhancer2_seq = Sequential()
    conv_enhancer2_seq.add(Convolution2D(1024, 4, 40, activation = 'relu', border_mode = 'valid',
                                        dim_ordering = 'th', input_shape = (1, 4, SEQ_LEN)))
    conv_enhancer2_seq.add(MaxPooling2D(pool_size = (1, 20), border_mode = 'valid', dim_ordering = 'th'))
    conv_enhancer2_seq.add(Reshape((1024, 98)))
    merged = Sequential()
    merged.add(Merge([conv_enhancer1_seq, conv_enhancer2_seq], mode = 'concat'))
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

def training(model):
    print('Loading data...')
	SEQ_LEN = 2000
    enhancer_shape = (-1, 1, SEQ_LEN, 4)
    seq1 = np.load(cell+'/enhancer1.npz')
    seq2 = np.load(cell+'/enhancer2.npz')
    #
    label = seq1['label'].shape[0]
    np.random.seed(label)
    rand_index = range(0, label)
    np.random.shuffle(rand_index)
    label = seq1['label'][rand_index]
    seq1 = seq1['sequence'].astype('float32').reshape(enhancer_shape).transpose(0, 1, 3, 2)[rand_index]
    seq2 = seq2['sequence'].astype('float32').reshape(enhancer_shape).transpose(0, 1, 3, 2)[rand_index]
    
    ## Train model
    model.compile(loss = 'binary_crossentropy',
                  optimizer = optimizers.Adam(lr = 0.00001),
                  metrics = ['acc', f1])
    filename = cell+'/models/best_model.h5'
    modelCheckpoint = ModelCheckpoint(filename, monitor = 'val_acc', save_best_only = True, mode = 'max')
    model.fit([seq1, seq2], label, nb_epoch = 50, batch_size = 100,
              validation_split = 0.1, callbacks = [modelCheckpoint])

""" MAIN """
cell = argv[1]
model = model_def()
training(model)
