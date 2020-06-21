
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
import torch.optim as optim


class Arguments():
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epoch = 20
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = True
        self.seed = 1
        self.log_interval = 200
        self.save_model = False

args = Arguments()


fname = 'Final_Flow.pkl'
data = pd.read_pickle(fname)
print('Data Loaded to Memory...')
window_sizes = [30, 40, 50, 70, 100]
LOG_INTERVAL = 5
BATCH_SIZE = 100
EPOCHS = 20

use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")



def train(model, device, federated_train_loader, optimizer, epoch):
    model.train()
    # Iterate through each gateway's dataset


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

    hook = sy.TorchHook(torch)
    torch.manual_seed(1)
    gatway1 = sy.VirtualWorker(hook, id="gatway1")
    gatway2 = sy.VirtualWorker(hook, id="gatway2")



    n_feature = X_train.shape[1]
    n_class = np.unique(y_train).shape[0]

    print("Number of training features : ", n_feature)
    print("Number of training classes : ", n_class)

    train_inputs = torch.tensor(X_train.astype(dtype='float64')).tag("#iot", "#network", "#data", "#train")
    train_labels = torch.tensor(y_train).tag("#iot", "#network", "#target", "#train")
    test_inputs = torch.tensor(X_test.astype(dtype='float64')).tag("#iot", "#network", "#data", "#test")
    test_labels = torch.tensor(y_test).tag("#iot", "#network", "#target", "#test")

    train_idx = int(len(train_labels) / 2)
    test_idx = int(len(test_labels) / 2)
    gatway1_train_dataset = sy.BaseDataset(train_inputs[:train_idx], train_labels[:train_idx]).send(gatway1)
    gatway2_train_dataset = sy.BaseDataset(train_inputs[train_idx:], train_labels[train_idx:]).send(gatway2)
    gatway1_test_dataset = sy.BaseDataset(test_inputs[:test_idx], test_labels[:test_idx]).send(gatway1)
    gatway2_test_dataset = sy.BaseDataset(test_inputs[test_idx:], test_labels[test_idx:]).send(gatway2)

    # Create federated datasets, an extension of Pytorch TensorDataset class
    federated_train_dataset = sy.FederatedDataset([gatway1_train_dataset, gatway2_train_dataset])
    federated_test_dataset = sy.FederatedDataset([gatway1_test_dataset, gatway2_test_dataset])

    # Create federated dataloaders, an extension of Pytorch DataLoader class
    federated_train_loader = sy.FederatedDataLoader(federated_train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    federated_test_loader = sy.FederatedDataLoader(federated_test_dataset, shuffle=False, batch_size=BATCH_SIZE)
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
        model.tain()
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        hist_RNN = rnn_model.train(model, device, federated_train_loader,optimizer, 20)

        with open('hist-' + model_name + "[" + str(window_size) + '].json', 'w') as f:
            json.dump(hist_RNN.history, f)
        model.save(model_name + "[" + str(window_size) + "].h5")



