import torch
import torch.nn as nn
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter

from se2seq import Encoder
from se2seq import Decoder
from se2seq import Seq2Seq
from deepl_baseline import DeepBaseline
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

    #Model hyperparameters
    lr = float(config['train']['lr'])
    num_epochs = int(config['train']['num_epochs'])
    batch_size = int(config['train']['batch_size'])
    num_layers = int(config['train']['num_layers'])
    algo = config['train']['algo']

    input_horizon = int(config['data']['input_horizon'])
    f_name = algo + '_' + str(input_horizon) + '.pth.tar'

    if algo == 'seq2seq':
        encoder = Encoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
        decoder = Decoder(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, output_size=output_size).to(device)
        model = Seq2Seq(encoder, decoder).to(device)
        criterion = nn.CrossEntropyLoss()
    elif algo == 'baseline':
        # ouput size = seq length
        model = DeepBaseline(input_size=input_size, hidden_size=hidden_size, output_size=Y_train.shape[1])
        criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    #writer = SummaryWriter('runs/seq2seq')
    #step = 0

    n_batches = int(X_train.shape[0] / batch_size)

    losses = []

    for epoch in range(num_epochs):

        print('Epoch : ' + str(epoch) + '/' + str(num_epochs))

        for b in range(n_batches):

            b = b*batch_size
            input_batch = X_train[b: b + batch_size, :, :]
            target_batch = Y_train[b: b + batch_size, :, :]
            target_label = target[b: b + batch_size, :]

            if algo == 'seq2seq':
                outputs = model(input_batch, target_batch)
                outputs = outputs.reshape(-1, outputs.shape[2])
                target_label = target_label.reshape(-1)
            elif algo == 'baseline':
                target_label = target_label.float()
                hidden = model.init_hidden(batch_size).to(device)
                outputs = model(input_batch, hidden)

            #print(target_label.shape, outputs.shape)

            optimizer.zero_grad()
            if algo == 'seq2seq':
                loss = criterion(outputs, target_label)
            elif algo == 'baseline':
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

    loss_file = algo + '_' + str(input_horizon) + '.pkl'
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

    # Model hyperparameters
    lr = float(config['train']['lr'])
    threshold = float(config['train']['threshold'])
    batch_size = int(config['train']['batch_size'])
    num_layers = int(config['train']['num_layers'])
    algo = config['train']['algo']

    if algo == 'seq2seq':
        encoder = Encoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
        decoder = Decoder(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, output_size=output_size).to(device)

        model = Seq2Seq(encoder, decoder).to(device)
    elif algo == 'baseline':
        model = DeepBaseline(input_size=input_size, hidden_size=hidden_size, output_size=Y_test.shape[1])

    optimizer = optim.Adam(model.parameters(), lr=lr)

    input_horizon = int(config['data']['input_horizon'])
    f_name = algo + '_' + str(input_horizon) + '.pth.tar'

    load_checkpoint(config, f_name, model, optimizer)

    model.eval()

    n_batches = int(X_test.shape[0] / batch_size)
    target_len = Y_test.shape[1]

    pred = list()
    target_ = list()

    print('Evaluaing...')

    for b in range(n_batches):

        b = b * batch_size
        input_batch = X_test[b: b + batch_size, :, :]
        target_batch = Y_test[b: b + batch_size, :, :]
        target_label = target[b: b + batch_size, :]

        if algo == 'seq2seq':
            prediction = np.zeros((batch_size, target_len, 1))
            outputs = model(input_batch, target_batch, 0.0)
            #outputs = outputs.reshape(-1, outputs.shape[2])
            #target_label = target_label.reshape(-1)

            for t in range(target_len):
                topv, topi = outputs[:, t].topk(1)
                prediction[:, t] = topi.cpu()
            pred.append(prediction)
        elif algo == 'baseline':
            hidden = model.init_hidden(batch_size).to(device)
            prediction = model(input_batch, hidden)
            prediction[prediction >= threshold] = 1
            prediction[prediction < threshold] = 0
            pred.append(prediction.detach().numpy())

        target_.append(target_label)

    #pred = np.array(pred)
    pred = np.array(pred).reshape(-1, target_len)
    target_ = np.array(target_).reshape(-1, target_len)
    print(pred.shape, target_.shape)
    f1 = f1_score(target_.ravel(), pred.ravel(), average=None)
    bal_acc = balanced_accuracy_score(target_.ravel(), pred.ravel())
    print(f1, bal_acc)
    return f1, bal_acc


def log_result(config, n_train, n_test, bal_acc, f1):

    result_path = config['result']['path']
    input_horizon = int(config['data']['input_horizon'])
    output_horizon = int(config['data']['output_horizon'])
    lr = float(config['train']['lr'])
    num_epochs = int(config['train']['num_epochs'])
    algo = config['train']['algo']
    threshold = float(config['train']['threshold'])

    result_row = [algo, n_train, n_test, input_horizon, output_horizon, 'Adam', lr, num_epochs, threshold, bal_acc, f1[0], f1[1]]

    result_file = os.path.join(result_path, algo + '.csv')

    if not os.path.isfile(result_file):
        header = ['Model', 'n_train', 'n_test', 'input_horizon', 'output_horizon', 'optim', 'lr', 'epoch', 'threshold',
                  'bal_acc', 'F1_0', 'F1_1']
        with open(result_file, "a+", newline='') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow(header)
            csv_writer.writerow(result_row)
    else:
        with open(result_file, "a+", newline='') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow(result_row)