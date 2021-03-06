import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
from data import Data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
#torch.set_deterministic(True)

class DeepBaseline(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):

        super(DeepBaseline, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        #hidden = torch.cat((hidden, features.unsqueeze(0)), 2)
        out = self.linear(hidden)
        out = out.squeeze(0)
        return torch.sigmoid(out)

    def init_hidden(self, batch_size):
        '''
        :param batch_size:      input.shape[0]
        :return:                zeroed initialized hidden states
        '''
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
