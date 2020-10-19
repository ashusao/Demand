import torch
import torch.nn as nn
import random
from data import Data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1):
        '''
        :param input_size: number of features in input
        :param hidden_size: number of features in hidden state
        :param num_layers: number of stacked layers
        '''
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, input, hidden):
        '''
        :param input:       input of shape (batch, seq_len, input_size)
        :param hidden:      initial hidden state (num_layers, batch_size, hidden_size)
        :return: output, hidden
                output gives all outputs in seq. shape (batch_size, seq_len, output_size)
                hidden represents context vector. shape (num_layers, batch_size, hidden_size)
        '''
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        '''
        :param batch_size:      input.shape[0]
        :return:                zeroed initialized hidden states
        '''
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        '''

        :param input_size:              number of features in input
        :param hidden_size:             number of features in hidden state
        :param output_size:             number of classes in output vector
        :param num_layers:              number of stacked layers
        '''
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        '''
        :param input:           should be 2D (batch_size, input_size)
        :param hidden:          last hidden state (num_layers, batch_size, hidden_size)
        :return:
                output          decoder output at time t (batch_size, output_size)
                hidden          last hidden state (num_layers, batch_size, hidden_size)
        '''
        # Add an extra dimension for seq_len = 1 because we are sending one input at a time
        output, hidden = self.gru(input.unsqueeze(1), hidden)
        output = self.linear(output)

        # squeeze the seq_len dimension so that output is (batch_size, output_dim)
        output = output.squeeze(1)
        return output, hidden

class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.data_obj = Data()

    def forward(self, source, target, teacher_force_ratio=0.8):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        output_size = target.shape[2]

        outputs = torch.zeros(batch_size, target_len, output_size).to(device)
        hidden = self.encoder.init_hidden(batch_size).to(device)

        encoder_out, hidden = self.encoder(source, hidden)
        #print(encoder_out.shape, hidden.shape)

        # First input to decoder will be last output of encoder
        decoder_input = source[:, -1, :] # shape(batch_size, input_size)
        #print(decoder_input.shape)

        use_teacher_force = True if random.random() < teacher_force_ratio else False

        if use_teacher_force:
            # feed the target as next input
            for t in range(target_len):
                # Using precious hidden state which is context from encoder at start
                out, hidden = self.decoder(decoder_input, hidden)
                outputs[:, t] = out

                decoder_input = target[:, t]
        else:
            # feed output as next input
            for t in range(target_len):
                out, hidden = self.decoder(decoder_input, hidden)
                outputs[:, t] = out

                topv, topi = out.topk(1)
                decoder_input = self.data_obj.one_hot_transform(topi.unsqueeze(1).detach().cpu())
                decoder_input = torch.from_numpy(decoder_input).float().squeeze(1).to(device) # squeeze the seq_len dim (batch_size, input_size)

        return outputs











