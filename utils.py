import os
import matplotlib.pyplot as plt
import torch
import pickle

plt.switch_backend('agg')
import matplotlib.ticker as ticker

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
#torch.set_deterministic(True)

def save_checkpoint(state, config, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    algo = config['train']['algo']

    if algo == 'seq2seq':
        model_dir = config['model']['seq2seq_path']
    elif algo == 'baseline':
        model_dir = config['model']['baseline_path']

    torch.save(state, os.path.join(model_dir, filename))


def load_checkpoint(config, filename, model, optimizer):
    print("=> Loading checkpoint")
    print(filename)
    algo = config['train']['algo']

    if algo == 'seq2seq':
        model_dir = config['model']['seq2seq_path']
    elif algo == 'baseline':
        model_dir = config['model']['baseline_path']

    checkpoint = torch.load(os.path.join(model_dir, filename), map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def save_loss(config, train_loss, f_name):
    algo = config['train']['algo']

    if algo == 'seq2seq':
        train_loss_dir = config['result']['seq2seq_train_loss_path']
        #test_loss_dir = config['result']['seq2seq_test_loss_path']
    elif algo == 'baseline':
        train_loss_dir = config['result']['baseline_train_loss_path']
        #test_loss_dir = config['result']['baseline_test_loss_path']

    with open(os.path.join(train_loss_dir, f_name), 'wb') as f:
        pickle.dump(train_loss, f)

    #with open(os.path.join(test_loss_dir, f_name), 'wb') as f:
    #    pickle.dump(test_loss, f)


def show_plot(config, precision, recall, ap, n_features, input_horizon):

    result_path = config['result']['path']
    #input_horizon = int(config['data']['input_horizon'])
    algo = config['train']['algo']

    file = algo + '_' + str(input_horizon) + '.png'
    result_file = os.path.join(result_path, algo, file)

    title = algo + '_' + str(input_horizon) + ' AP=' + str(ap)
    #plt.figure()
    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.plot(recall, precision, marker='o', linewidth=1)
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.set_title(title)
    #plt.show()
    fig.savefig(result_file)
