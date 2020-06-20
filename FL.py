
import part12 as  rnn_model

from keras.layers import Dense , Activation

from keras.layers import LSTM,GRU

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import json
import torch
import syft as sy




fname = 'Final_Flow.pkl'
data = pd.read_pickle(fname)
print('Data Loaded to Memory...')
window_sizes = [30, 40, 50, 70, 100]



for window_size in window_sizes:
    models = {}
    labels = []
    samples = []
    for k in data.keys():
        d = data[k]
        if len(d[1]) > window_size - 1:
            sample = d[1][0:window_size]
            label = d[0]
            sample = np.array(sample).T
            # print(np.shape(sample))
            samples.append(sample)
            labels.append(label)
    print('Data Generated For Window Size  ' + str(window_size))
    data_size = 79
    n_class = len(set(labels))
    labels = rnn_model.one_hot(labels)
    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=.2, random_state=1)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print('X Shape:', np.shape(X_train))
    print('Y Shape:', np.shape(y_train))
    #models = []
    # Model 1
    model_SLSTM1 = Sequential()
    model_SLSTM1.add(GRU(units=100, return_sequences=True, input_shape=(data_size, window_size)))
    model_SLSTM1.add(GRU(units=50, return_sequences=True))
    model_SLSTM1.add(GRU(units=20, return_sequences=False))
    model_SLSTM1.add(Dense(units=n_class))
    model_SLSTM1.add(Activation('softmax'))
    model_SLSTM1.compile(loss='categorical_crossentropy', optimizer='adam',
                         metrics=['categorical_accuracy', rnn_model.precision, rnn_model.recall, rnn_model.f1])
    model_SLSTM1.summary()
    models['LSTM-100-50-20'] = model_SLSTM1
    # Model 2
    model_SLSTM2 = Sequential()
    model_SLSTM2.add(GRU(units=100, return_sequences=True, input_shape=(data_size, window_size)))
    model_SLSTM2.add(GRU(units=100, return_sequences=True))
    model_SLSTM2.add(GRU(units=100, return_sequences=False))
    model_SLSTM2.add(Dense(units=n_class))
    model_SLSTM2.add(Activation('softmax'))
    model_SLSTM2.compile(loss='categorical_crossentropy', optimizer='adam',
                         metrics=['categorical_accuracy', rnn_model.precision, rnn_model.recall, rnn_model.f1])
    model_SLSTM2.summary()
    models['LSTM-100-100-100'] = model_SLSTM2
    # Model 3
    model_SLSTM3 = Sequential()
    model_SLSTM3.add(GRU(units=100, return_sequences=True, input_shape=(data_size, window_size)))
    model_SLSTM3.add(GRU(units=100, return_sequences=False))
    model_SLSTM3.add(Dense(units=n_class))
    model_SLSTM3.add(Activation('softmax'))
    model_SLSTM3.compile(loss='categorical_crossentropy', optimizer='adam',
                         metrics=['categorical_accuracy', rnn_model.precision, rnn_model.recall, rnn_model.f1])
    model_SLSTM3.summary()
    models['LSTM-100-100'] = model_SLSTM3
    # Model 4
    model_SLSTM4 = Sequential()
    model_SLSTM4.add(GRU(units=200, return_sequences=True, input_shape=(data_size, window_size)))
    model_SLSTM4.add(GRU(units=100, return_sequences=False))
    model_SLSTM4.add(Dense(units=n_class))
    model_SLSTM4.add(Activation('softmax'))
    model_SLSTM4.compile(loss='categorical_crossentropy', optimizer='adam',
                         metrics=['categorical_accuracy', rnn_model.precision, rnn_model.recall, rnn_model.f1])
    model_SLSTM4.summary()
    models['LSTM-200-100'] = model_SLSTM4
    # Model 5
    model_SLSTM5 = Sequential()
    model_SLSTM5.add(GRU(units=200, return_sequences=False, input_shape=(data_size, window_size)))
    model_SLSTM5.add(Dense(units=n_class))
    model_SLSTM5.add(Activation('softmax'))
    model_SLSTM5.compile(loss='categorical_crossentropy', optimizer='adam',
                         metrics=['categorical_accuracy', rnn_model.precision, rnn_model.recall, rnn_model.f1])
    model_SLSTM5.summary()
    models['LSTM-200'] = model_SLSTM5
    # Model 6
    model_SLSTM6 = Sequential()
    model_SLSTM6.add(GRU(units=100, return_sequences=False, input_shape=(data_size, window_size)))
    model_SLSTM6.add(Dense(units=n_class))
    model_SLSTM6.add(Activation('softmax'))
    model_SLSTM6.compile(loss='categorical_crossentropy', optimizer='adam',
                         metrics=['categorical_accuracy', rnn_model.precision, rnn_model.recall, rnn_model.f1])
    model_SLSTM6.summary()
    models['LSTM-100'] = model_SLSTM6

    for model_name in models.keys():
        model = models[model_name]
        hist_RNN = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=5000,
                             verbose=1)  # 80 train 20 test
        with open('hist-' + model_name + "[" + str(window_size) + '].json', 'w') as f:
            json.dump(hist_RNN.history, f)
        model.save(model_name + "[" + str(window_size) + "].h5")



