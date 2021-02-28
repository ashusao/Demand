import torch
import torch.nn as nn
import torch.optim as optim

from se2seq import Encoder
from se2seq import Decoder
from se2seq import Seq2Seq
from se2seq import Embedding
from se2seq import AttnDecoder
from deepl_baseline import DeepBaseline

from utils import load_checkpoint
from utils import save_checkpoint
from data import Data

from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from loss import FocalLoss

import numpy as np
import os
import csv
import sys

from utils import save_loss
from utils import show_plot

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
#torch.set_deterministic(True) # type: ignore


def compute_weights(targets):

    positive = torch.zeros(1, dtype=torch.float, device=device)
    negative = torch.zeros(1, dtype=torch.float, device=device)
    for i in torch.arange(0, targets.shape[0]):
        t = targets[i]
        pos = (t == 1).sum()
        neg = (t == 0).sum()
        positive += pos
        negative += neg

    high = positive if positive > negative else negative
    p_w = high.float() / positive.float()
    n_w = high.float() / negative.float()
    return p_w, n_w

def compute_weight_matrix(targets, positive_weight, negative_weight):
    """
        :param targets: targets is a 2d target data batch_size x seq_len
        :return weight: 2d weight matrix containing weight matrix corresponding to each label
        """
    weights = torch.tensor((), dtype=torch.float, device=device)
    weights = weights.new_zeros(targets.size())

    for i in torch.arange(0, targets.shape[0]):
        t = targets[i]
        weights[i, t == 1] = positive_weight
        weights[i, t == 0] = negative_weight

    return weights

def train(config, X_train, Y_train, X_test, Y_test, Train_cs_features, Test_cs_features,
          Train_spatial_features, Test_spatial_features, Train_pattern_features, Test_pattern_features):

    Y_train = torch.from_numpy(Y_train).float()
    X_train = torch.from_numpy(X_train).float()

    Train_cs_features = torch.from_numpy(Train_cs_features).float()
    Test_cs_features = torch.from_numpy(Test_cs_features).float()

    Train_spatial_features = torch.from_numpy(Train_spatial_features).float()
    Test_spatial_features = torch.from_numpy(Test_spatial_features).float()

    Train_pattern_features = torch.from_numpy(Train_pattern_features).float()
    Test_pattern_features = torch.from_numpy(Test_pattern_features).float()

    Y_test = torch.from_numpy(Y_test).float()
    X_test = torch.from_numpy(X_test).float()

    if len(X_train.shape) == 2: # if 2d make it 3d
        X_train = X_train.unsqueeze(2)  # add 3rd dimesion when not one hot enocded or no additional features
        X_test = X_test.unsqueeze(2)

    input_size = X_train.shape[2] # 1 or additional attributes
    output_size = 1
    #output_size = X_train.shape[2]
    hidden_size = int(config['train']['hidden_size'])
    embed_size = hidden_size

    #Model hyperparameters
    lr = float(config['train']['lr'])
    dropout = float(config['train']['dropout'])
    wd = float(config['train']['wd'])
    num_epochs = int(config['train']['num_epochs'])
    batch_size = int(config['train']['batch_size'])
    num_layers = int(config['train']['num_layers'])
    algo = config['train']['algo']
    decode = config['model']['decoder']
    feat = config.getboolean('data', 'features')

    input_horizon = int(config['data']['input_horizon'])
    f_name = algo + '_' + str(input_horizon) + '.pth.tar'

    if algo == 'seq2seq':
        encoder = Encoder(input_size=input_size, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers).to(device)
        embedding_cs = Embedding(feat_size=Train_cs_features.shape[1], embed_size=embed_size)
        embedding_spatial = Embedding(feat_size=Train_spatial_features.shape[1], embed_size=embed_size)
        embedding_pattern = Embedding(feat_size=Train_pattern_features.shape[1], embed_size=embed_size)
        embedding = Embedding(feat_size=hidden_size + embed_size, embed_size=embed_size)
        #embedding = Embedding(feat_size=Train_cs_features.shape[1] + hidden_size, embed_size=embed_size)

        if decode == 'attention':
            decoder = AttnDecoder(input_size=1, hidden_size=hidden_size, output_size=output_size, input_len=X_train.shape[1],
                                  feat_size_cs=Train_cs_features.shape[1], feat_size_spatial=Train_spatial_features.shape[1],
                                  dropout=dropout, num_layers=num_layers).to(device)

        if decode == 'decoder':
            decoder = Decoder(input_size=1, hidden_size=hidden_size, output_size=output_size,
                              feat_size_cs=Train_cs_features.shape[1], feat_size_spatial=Train_spatial_features.shape[1],
                              dropout=dropout, num_layers=num_layers).to(device)
            '''decoder = Decoder(input_size=X_train.shape[2], hidden_size=hidden_size, output_size=output_size,
                              feat_size_cs=Train_cs_features.shape[1],
                              feat_size_spatial=Train_spatial_features.shape[1],
                              dropout=dropout, num_layers=num_layers).to(device)'''

        if decode == 'features':
            decoder = Decoder(input_size=1, hidden_size=hidden_size, output_size=output_size,
                              feat_size_cs=Train_cs_features.shape[1], feat_size_spatial=Train_spatial_features.shape[1],
                              dropout=dropout, num_layers=num_layers).to(device)
            '''decoder = Decoder(input_size=1, hidden_size=hidden_size + Train_pattern_features.shape[1], output_size=output_size, feat_size_cs=Train_cs_features.shape[1],
                              feat_size_spatial=Train_spatial_features.shape[1],
                              dropout=dropout, num_layers=num_layers).to(device)'''

        model = Seq2Seq(encoder, decoder, embedding_cs, embedding_spatial, embedding_pattern, embedding, config).to(device)
    elif algo == 'baseline':
        # ouput size = seq length
        model = DeepBaseline(input_size=input_size, hidden_size=hidden_size, output_size=Y_train.shape[1]).to(device)

    criterion = nn.BCELoss()
    #criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total trainable Params: ', pytorch_total_params)

    n_batches_train = int(X_train.shape[0] / batch_size)
    n_batches_test = int(X_test.shape[0] / batch_size)
    print(n_batches_train, n_batches_test)
    #positive_wt, negative_wt = compute_weights(Y_train)

    train_loss = []
    test_loss = []
    phases = ['train', 'test']

    for epoch in range(num_epochs):

        print('Epoch : ' + str(epoch) + '/' + str(num_epochs))

        for b in range(n_batches_train):

            b = b*batch_size
            for phase in phases:

                if phase == 'train':
                    input_batch = X_train[b: b + batch_size, :, :].to(device)
                    target_label = Y_train[b: b + batch_size, :].to(device)   # here
                    #target_label = Y_train[b: b + batch_size, :, :].to(device)  # here
                    features_cs = Train_cs_features[b: b + batch_size, :].to(device)
                    features_spatial = Train_spatial_features[b: b + batch_size, :].to(device)
                    features_pattern = Train_pattern_features[b: b + batch_size, :].to(device)
                    #positive_wt, negative_wt = compute_weights(target_label)
                    model.train()
                else:
                    input_batch = X_test[b % X_test.shape[0]: ((b % X_test.shape[0]) + batch_size), :, :].to(device)
                    target_label = Y_test[b % Y_test.shape[0]: ((b % Y_test.shape[0]) + batch_size), :].to(device)  # here
                    #target_label = Y_test[b % Y_test.shape[0]: ((b % Y_test.shape[0]) + batch_size), :, :].to(device)  # here
                    features_cs = Test_cs_features[b % Test_cs_features.shape[0]: ((b % Test_cs_features.shape[0]) + batch_size), :].to(device)
                    features_spatial = Test_spatial_features[b % Test_spatial_features.shape[0]: ((b % Test_spatial_features.shape[0]) + batch_size), :].to(device)
                    features_pattern = Test_pattern_features[b % Test_pattern_features.shape[0]: ((b % Test_pattern_features.shape[0]) + batch_size), :].to(device)
                    #positive_wt, negative_wt = compute_weights(target_label)
                    model.eval()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if algo == 'seq2seq':
                        outputs = model(input_batch, target_label, features_cs, features_spatial, features_pattern, 0.0) # no teacher force
                    elif algo == 'baseline':
                        hidden = model.init_hidden(batch_size).to(device)
                        outputs = model(input_batch, hidden)

                    #print(target_label.shape, outputs.shape)

                    #weights = compute_weight_matrix(target_label, positive_wt, negative_wt)
                    #criterion.weight = weights

                    loss = criterion(outputs, target_label)  # here
                    print(loss)

                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                        optimizer.step()
                        train_loss.append(loss.item())
                    else:
                        test_loss.append(loss.item())

        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint, config, filename=f_name)

    loss_file = algo + '_' + str(input_horizon) + '.pkl'
    save_loss(config, train_loss, test_loss, loss_file)
    print('Finished training')


def evaluate(config, X_test, Y_test, Test_cs_features, Test_spatial_features, Test_pattern_features, n_train):
    '''
    :param config:
    :param X_test: one hot transform X_test
    :param Y_test: one hot transform Y_test
    :param target: 2d Y_test
    :return:
    '''

    Y_test = torch.from_numpy(Y_test).float()
    X_test = torch.from_numpy(X_test).float()
    Test_cs_features = torch.from_numpy(Test_cs_features).float()
    Test_spatial_features = torch.from_numpy(Test_spatial_features).float()
    Test_pattern_features = torch.from_numpy(Test_pattern_features).float()

    if len(X_test.shape) == 2:
        X_test = X_test.unsqueeze(2)  # add 3rd dimension when not one hot encoded and no additional features

    n_test = X_test.shape[0]

    input_size = X_test.shape[2]
    output_size = 1
    #output_size = X_test.shape[2]
    hidden_size = int(config['train']['hidden_size'])
    embed_size = hidden_size

    # Model hyperparameters
    lr = float(config['train']['lr'])
    dropout = float(config['train']['dropout'])
    wd = float(config['train']['wd'])
    batch_size = int(config['train']['batch_size'])
    num_layers = int(config['train']['num_layers'])
    algo = config['train']['algo']
    decode = config['model']['decoder']
    feat = config.getboolean('data', 'features')

    if algo == 'seq2seq':
        encoder = Encoder(input_size=input_size, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers).to(device)
        embedding_cs = Embedding(feat_size=Test_cs_features.shape[1], embed_size=embed_size)
        embedding_spatial = Embedding(feat_size=Test_spatial_features.shape[1], embed_size=embed_size)
        embedding_pattern = Embedding(feat_size=Test_pattern_features.shape[1], embed_size=embed_size)
        embedding = Embedding(feat_size=hidden_size + embed_size, embed_size=embed_size)
        #embedding = Embedding(feat_size=Test_cs_features.shape[1] + hidden_size, embed_size=embed_size)

        if decode == 'attention':
            decoder = AttnDecoder(input_size=1, hidden_size=hidden_size, output_size=output_size, input_len=X_test.shape[1],
                                  feat_size_cs=Test_cs_features.shape[1],  feat_size_spatial=Test_spatial_features.shape[1],
                                  dropout=dropout, num_layers=num_layers).to(device)

        if decode == 'decoder':
            decoder = Decoder(input_size=1, hidden_size=hidden_size, output_size=output_size,
                              feat_size_cs=Test_cs_features.shape[1],  feat_size_spatial=Test_spatial_features.shape[1],
                              dropout=dropout, num_layers=num_layers).to(device)
            '''decoder = Decoder(input_size=X_test.shape[2], hidden_size=hidden_size, output_size=output_size,
                              feat_size_cs=Test_cs_features.shape[1], feat_size_spatial=Test_spatial_features.shape[1],
                              dropout=dropout, num_layers=num_layers).to(device)'''

        if decode == 'features':
            decoder = Decoder(input_size=1, hidden_size=hidden_size, output_size=output_size,
                              feat_size_cs=Test_cs_features.shape[1],  feat_size_spatial=Test_spatial_features.shape[1],
                              dropout=dropout, num_layers=num_layers).to(device)
            '''decoder = Decoder(input_size=1, hidden_size=hidden_size + Test_pattern_features.shape[1], output_size=output_size, feat_size_cs=Test_cs_features.shape[1],
                              feat_size_spatial=Test_spatial_features.shape[1],
                              dropout=dropout, num_layers=num_layers).to(device)'''

        model = Seq2Seq(encoder, decoder, embedding_cs, embedding_spatial, embedding_pattern, embedding, config).to(device)
    elif algo == 'baseline':
        model = DeepBaseline(input_size=input_size, hidden_size=hidden_size, output_size=Y_test.shape[1]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

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
        input_batch = X_test[b: b + batch_size, :, :].to(device)
        target_label = Y_test[b: b + batch_size, :].to(device)  #here
        #target_label = Y_test[b: b + batch_size, :, :].to(device)  # here
        features_cs = Test_cs_features[b: b + batch_size, :].to(device)
        features_spatial = Test_spatial_features[b: b + batch_size, :].to(device)
        features_pattern = Test_pattern_features[b: b + batch_size, :].to(device)

        if algo == 'seq2seq':
            # prediction is sigmoid activation
            prediction = model(input_batch, target_label, features_cs, features_spatial, features_pattern, 0.0)
            pred.append(prediction.detach().cpu().numpy())
        elif algo == 'baseline':
            # prediction is sigmoid activation
            hidden = model.init_hidden(batch_size).to(device)
            prediction = model(input_batch, hidden)
            pred.append(prediction.detach().cpu().numpy())

        target_.append(target_label.detach().cpu().numpy()) #here

    pred = np.array(pred)
    target_ = np.array(target_)
    print(pred.shape, target_.shape)
    #pred = pred.reshape(-1, target_len)
    #target_ = target_.reshape(-1, target_len)
    #print(pred.shape, target_.shape)
    #print(np.unique(target_.ravel()))

    eval_tests = config.getboolean('data', 'eval_tests')

    if not eval_tests:
        log_plot(config, n_train, n_test, X_test.shape[2], pred, target_)

    return pred, target_


def log_plot(config, n_train, n_test, n_features, prediction, target):

    prec, rec, th = precision_recall_curve(target.ravel(), prediction.ravel())
    ap = average_precision_score(target.ravel(), prediction.ravel())
    print('threshold: ')
    print(th)
    show_plot(config, prec, rec, ap, n_features)

    fscore = (2 * prec * rec) / (prec + rec)
    fscore = np.nan_to_num(fscore)
    ix = np.argmax(fscore)
    print('Best Threshold=%f, F-Score=%.3f' % (th[ix], fscore[ix]))

    log_result(config, n_train, n_test, th[ix], prediction, target)


def log_result(config, n_train, n_test, th, prediction, target):

    result_path = config['result']['path']
    input_horizon = int(config['data']['input_horizon'])
    output_horizon = int(config['data']['output_horizon'])
    lr = float(config['train']['lr'])
    num_epochs = int(config['train']['num_epochs'])
    algo = config['train']['algo']
    comment = config['result']['comment']

    #thresholds = np.arange(0.1, 1, 0.1)
    result_rows = list()

    #for th in thresholds:
    pred = np.copy(prediction)
    pred[pred >= th] = 1
    pred[pred < th] = 0
    f1 = f1_score(target.ravel(), pred.ravel(), average=None)
    bal_acc = balanced_accuracy_score(target.ravel(), pred.ravel())
    result_row = [algo, n_train, n_test, input_horizon, output_horizon, 'Adam', lr, num_epochs, th, bal_acc, f1[0], f1[1], comment]
    result_rows.append(result_row)

    result_file = os.path.join(result_path, algo + '.csv')

    if not os.path.isfile(result_file):
        header = ['Model', 'n_train', 'n_test', 'input_horizon', 'output_horizon', 'optim', 'lr', 'epoch', 'threshold',
                  'bal_acc', 'F1_0', 'F1_1', 'comment']

        with open(result_file, "a+", newline='') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow(header)
            csv_writer.writerows(result_rows)
    else:
        with open(result_file, "a+", newline='') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerows(result_rows)