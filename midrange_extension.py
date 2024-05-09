# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:55:14 2024

@author: duttahr1
"""




import tensorflow as tf
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from tensorflow.keras.optimizers import SGD, Adagrad
from random import seed
from random import randint
import math
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import copy
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

k_for_kfold = 10


kfold = KFold(k_for_kfold, True)





dataset = loadtxt("3_finger_mr.csv", dtype = str, delimiter=',')

acc_x = np.reshape(dataset[3:,2].astype(float),(-1,1))
acc_y = np.reshape(dataset[3:,3].astype(float),(-1,1))
acc_z = np.reshape(dataset[3:,4].astype(float),(-1,1))

ang_x = np.reshape(dataset[3:,7].astype(float),(-1,1))
ang_y = np.reshape(dataset[3:,8].astype(float),(-1,1))
ang_z = np.reshape(dataset[3:,9].astype(float),(-1,1))


me_class = np.zeros((219,901))+1


ind = 0
i=0
for x in range(219):
    
    me_class[ind,0:150]=acc_x[i:i+150].T
    me_class[ind,150:300]=acc_y[i:i+150].T
    me_class[ind,300:450]=acc_z[i:i+150].T
    me_class[ind,450:600]=ang_x[i:i+150].T
    me_class[ind,600:750]=ang_y[i:i+150].T
    me_class[ind,750:900]=ang_z[i:i+150].T
    ind+=1
    i = i+150
    
    


per = 3
# declaring new list
mr_class = np.zeros((219,301))+1

 
# looping over array
for j in range(219):
    cntr = 0
    ind = 0
    for i in range(901):
        if(cntr % per == 0):
            mr_class[j,ind]=me_class[j,i]
            ind += 1
        # incrementing counter
        cntr += 1
        


    
dataset = loadtxt("3_finger_fe.csv", dtype = str, delimiter=',')


full_data = shuffle(dataset)

n_features = np.size(full_data,1)-1


scaler = MinMaxScaler()
# transform data
full_data[:,0:n_features] = scaler.fit_transform(full_data[:,0:n_features])



train_x, test_x, train_y, test_y = train_test_split(full_data[:,0:n_features],full_data[:,n_features], test_size = 0.1, random_state = 0)


model = Sequential()
model.add(Dense(500, input_dim = n_features, activation = 'relu'))
model.add(Dense(300, activation = 'relu'))
model.add(Dense(200, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(train_x, train_y, epochs = 300, batch_size = 5)
scores = model.evaluate(test_x, test_y)


kk = model.predict(full_data[:,0:n_features])






























# shuff_data_tr = []
# shuff_data_test = []

# for train, test in kfold.split(full_data):
#     shuff_data_tr.append(full_data[train])
#     shuff_data_test.append(full_data[test])


# hist = []
# scr_tr = []
# scr_va = []

# for i in range(k_for_kfold):
    
#     print('Run:',i)
    
#     history = 0
    
       
#     model = Sequential()

#     model.add(Dense(500, input_dim=300, activation='relu'))
#     model.add(Dense(500, activation='relu'))
#     # model.add(Dense(500, activation='relu'))
#     model.add(Dense(400, activation='relu'))
#     model.add(Dense(200, activation='relu'))
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(50, activation='relu'))
#     # model.add(Dense(25, activation='relu'))
#     model.add(Dense(1, activation='softmax'))
    
#     sgd = SGD(lr=0.01, momentum=0.9)
#     # adagrad = Adagrad(lr=0.01,initial_accumulator_value=0.1,
#     # epsilon=1e-07,
#     # name="Adagrad")
#     # model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy',tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.TruePositives(),tf.keras.metrics.FalseNegatives()])
#     model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    
#     X_tr = shuff_data_tr[i][:,0:300]
#     y_tr = shuff_data_tr[i][:,300]
#     X_va = shuff_data_test[i][:,0:300]
#     y_va = shuff_data_test[i][:,300]
#     # X_tr = np.concatenate((shuff_data_tr[i][:,0:75],shuff_data_tr[i][:,150:225]),axis=1)
#     # y_tr = shuff_data_tr[i][:,225]
#     # X_va = np.concatenate((shuff_data_test[i][:,0:75],shuff_data_test[i][:,150:225]),axis=1)
#     # y_va = shuff_data_test[i][:,225]
#     history=model.fit(X_tr, y_tr, validation_data=(X_va,y_va), epochs=300, batch_size=10,verbose=0)
#     hist.append(history)
#     scr_tr.append(model.evaluate(X_tr, y_tr, verbose=0))
#     scr_va.append(model.evaluate(X_va, y_va, verbose=0))
    
    
# #7,21,75

# true_p = []   

# false_p = []

# for i in range(k_for_kfold):
#     false_p.append(scr_va[i][3]/(scr_va[i][2]+scr_va[i][3]))
#     true_p.append(scr_va[i][4]/(scr_va[i][4]+scr_va[i][5]))





# print('True_positive:',sum(true_p)/k_for_kfold)
# print('False_positive:',sum(false_p)/k_for_kfold)
