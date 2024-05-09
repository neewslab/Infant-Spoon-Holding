# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:51:57 2024

@author: duttahr1
"""


import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adagrad
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler


k_for_kfold = 5


kfold = KFold(k_for_kfold, True)



dataset_raw = np.load("20240204_Pencil/Processed Data/fingertip_vs_overhand_ulnar_labelled.npy")

# dataset=np.concatenate((dataset_raw[:,2:3],dataset_raw[:,8:9],dataset_raw[:,10:11],dataset_raw[:,11:12],dataset_raw[:,12:13]),axis=1)

# dataset=dataset_raw[:,:3]
dataset=np.concatenate((dataset_raw[:,:6],dataset_raw[:,9:10]),axis = 1)
n_features = np.size(dataset,1)-1


scaler = MinMaxScaler()
# transform data
dataset[:,0:n_features+1] = scaler.fit_transform(dataset[:,0:n_features+1])

shuff_data_tr = []
shuff_data_test = []

for train, test in kfold.split(dataset):
    shuff_data_tr.append(dataset[train])
    shuff_data_test.append(dataset[test])


hist = []
scr_tr = []
scr_va = []

for i in range(k_for_kfold):
    
    
    history = 0
    
       
    model = Sequential()

    model.add(Dense(30, input_dim=n_features, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(40, activation='relu'))
    # model.add(Dense(100, activation='relu'))
    # model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    sgd = SGD(lr=0.01, momentum=0.9)
    adagrad = Adagrad(lr=0.01,initial_accumulator_value=0.1,
    epsilon=1e-07,
    name="Adagrad")
    # model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy',tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.TruePositives(),tf.keras.metrics.FalseNegatives()])
    model.compile(loss='MSE', optimizer='adam', metrics=['accuracy',tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.TruePositives(),tf.keras.metrics.FalseNegatives()])
    
    
    X_tr = shuff_data_tr[i][:,:n_features]
    y_tr = shuff_data_tr[i][:,n_features]
    X_va = shuff_data_test[i][:,:n_features]
    y_va = shuff_data_test[i][:,n_features]
    # X_tr = np.concatenate((shuff_data_tr[i][:,0:75],shuff_data_tr[i][:,150:225]),axis=1)
    # y_tr = shuff_data_tr[i][:,225]
    # X_va = np.concatenate((shuff_data_test[i][:,0:75],shuff_data_test[i][:,150:225]),axis=1)
    # y_va = shuff_data_test[i][:,225]
    history=model.fit(X_tr, y_tr, validation_data=(X_va,y_va), epochs=100, batch_size=10,verbose=0)
    hist.append(history)
    scr_tr.append(model.evaluate(X_tr, y_tr, verbose=0))
    scr_va.append(model.evaluate(X_va, y_va, verbose=0))
    
    
    print('Run:',i)
    
    
#7,21,75

true_p = []   

false_p = []

for i in range(k_for_kfold):
    false_p.append(scr_va[i][3]/(scr_va[i][2]+scr_va[i][3]))
    true_p.append(scr_va[i][4]/(scr_va[i][4]+scr_va[i][5]))





print('True_positive:',sum(true_p)/k_for_kfold)
print('False_positive:',sum(false_p)/k_for_kfold)