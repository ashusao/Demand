import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from torch.utils.tensorboard import SummaryWriter
from se2seq import Encoder
from se2seq import Decoder
from se2seq import Seq2Seq
from utils import load_checkpoint
from utils import save_checkpoint
from data import Data
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import os
import csv
from utils import save_loss

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def train(config, X_train, Y_train, target):

    target = torch.from_numpy(target).long().to(device)
    X_train = torch.from_numpy(X_train).float().to(device)
    Y_train = torch.from_numpy(Y_train).float().to(device)

    input_size = X_train.shape[2]
    output_size = Y_train.shape[2]
    hidden_size = 100
    num_layers = 1

    #Model hyperparameters
    lr = float(config['train']['lr'])
    num_epochs = int(config['train']['num_epochs'])
    batch_size = int(config['train']['batch_size'])

    input_horizon = int(config['data']['input_horizon'])
    f_name = 'seq2seq_' + str(input_horizon) + '.pth.tar'

    encoder = Encoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
    decoder = Decoder(input_size=input_size, hidden_size=hidden_size,
                      num_layers=num_layers, output_size=output_size).to(device)

    model = Seq2Seq(encoder, decoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    #writer = SummaryWriter('runs/seq2seq')
    #step = 0

    n_batches = int(X_train.shape[0] / batch_size)

    losses = []
    loss = 0

    for epoch in range(num_epochs):

        print('Epoch : ' + str(epoch) + '/' + str(num_epochs))

        for b in range(n_batches):

            b = b*batch_size
            input_batch = X_train[b: b + batch_size, :, :]
            target_batch = Y_train[b: b + batch_size, :, :]
            target_label = target[b: b + batch_size, :]

            outputs = model(input_batch, target_batch)
            outputs = outputs.reshape(-1, outputs.shape[2])
            target_label = target_label.reshape(-1)

            #print(target_label.shape, outputs.shape)

            optimizer.zero_grad()
            loss = criterion(outputs, target_label)
            print(loss)

            loss.backward()
            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()

            losses.append(loss.item())

        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint, config, filename=f_name)

    loss_file = 'seq2seq_' + str(input_horizon) + '.pkl'
    save_loss(config, losses, loss_file)
    print('Finished training')


def evaluate(config, X_test, Y_test, target):
    '''
    :param config:
    :param X_test: one hot transform X_test
    :param Y_test: one hot transform Y_test
    :param target: 2d Y_test
    :return:
    '''

    Y_test = torch.from_numpy(Y_test).long().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)

    input_size = X_test.shape[2]
    output_size = X_test.shape[2]
    hidden_size = 100
    num_layers = 1

    # Model hyperparameters
    lr = float(config['train']['lr'])
    batch_size = int(config['train']['batch_size'])

    encoder = Encoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
    decoder = Decoder(input_size=input_size, hidden_size=hidden_size,
                      num_layers=num_layers, output_size=output_size).to(device)

    model = Seq2Seq(encoder, decoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    input_horizon = int(config['data']['input_horizon'])
    f_name = 'seq2seq_' + str(input_horizon) + '.pth.tar'

    load_checkpoint(config, f_name, model, optimizer)

    model.eval()

    n_batches = int(X_test.shape[0] / batch_size)
    target_len = Y_test.shape[1]

    pred = list()
    target_ = list()

    for b in range(n_batches):

        prediction = np.zeros((batch_size, target_len, 1))

        b = b * batch_size
        input_batch = X_test[b: b + batch_size, :, :]
        target_batch = Y_test[b: b + batch_size, :, :]
        target_label = target[b: b + batch_size, :]

        outputs = model(input_batch, target_batch, 0.0)
        #outputs = outputs.reshape(-1, outputs.shape[2])
        #target_label = target_label.reshape(-1)

        for t in range(target_len):
            topv, topi = outputs[:, t].topk(1)
            prediction[:, t] = topi.cpu()

        pred.append(prediction)
        target_.append(target_label)

    pred = np.array(pred).reshape(-1, target_len)
    target_ = np.array(target_).reshape(-1, target_len)
    print(pred.shape, target_.shape)
    f1 = f1_score(target_.ravel(), pred.ravel())
    bal_acc = balanced_accuracy_score(target_.ravel(), pred.ravel())
    print(f1, bal_acc)
    return f1, bal_acc


def log_result(config, model, n_train, n_test, bal_acc, f1):

    result_path = config['result']['path']
    input_horizon = int(config['data']['input_horizon'])
    output_horizon = int(config['data']['output_horizon'])
    lr = float(config['train']['lr'])
    num_epochs = int(config['train']['num_epochs'])

    result_row = [model, n_train, n_test, input_horizon, output_horizon, 'Adam', lr, num_epochs, bal_acc, f1]

    result_file = os.path.join(result_path, 'DeepL.csv')

    if not os.path.isfile(result_file):
        header = ['Model', 'n_train', 'n_test', 'input_horizon', 'output_horizon', 'optim', 'lr', 'epoch', 'bal acc', 'F1']
        with open(result_file, "a+", newline='') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow(header)
            csv_writer.writerow(result_row)
    else:
        with open(result_file, "a+", newline='') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow(result_row)