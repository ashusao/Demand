import os
import matplotlib.pyplot as plt
import torch
import pickle

plt.switch_backend('agg')
import matplotlib.ticker as ticker


def save_checkpoint(state, config, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    model_dir = config['model']['seq2seq_path']
    torch.save(state, os.path.join(model_dir, filename))


def load_checkpoint(config, filename, model, optimizer):
    print("=> Loading checkpoint")
    model_dir = config['model']['seq2seq_path']
    checkpoint = torch.load(os.path.join(model_dir, filename))
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def save_loss(config, loss, f_name):
    loss_dir = config['result']['loss_path']
    with open(os.path.join(loss_dir, f_name), 'wb') as f:
        pickle.dump(loss, f)


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()
