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

        out = self.linear(output)

        # squeeze the seq_len dimension so that output is (batch_size, output_dim)
        out = out.squeeze(1)
        return torch.sigmoid(out), hidden

class Embedding(nn.Module):

    def __init__(self, feat_size, embed_size):
        super(Embedding, self).__init__()
        self.feat_size = feat_size
        self.embed_size = embed_size
        self.linear = nn.Linear(feat_size, embed_size)

    def forward(self, features):
        return self.linear(features)


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, embedding):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding
        self.data_obj = Data()

    def forward(self, source, target, features, teacher_force_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        #output_size = target.shape[2]

        #outputs = torch.zeros(batch_size, target_len, output_size).to(device)
        outputs = torch.zeros(batch_size, target_len).to(device)

        hidden = self.encoder.init_hidden(batch_size).to(device)

        encoder_out, hidden = self.encoder(source, hidden)
        features = self.embedding(features)

        # print(encoder_out.shape, hidden.shape, fetaures.shape)
        features = features.unsqueeze(0)
        hidden = torch.cat((hidden, features), 2)

        # First input to decoder will be last input of encoder
        #decoder_input = source[:, -1, :] # shape(batch_size, input_size)
        # input the state of charger without features
        decoder_input = source[:, -1, 0]  # [0] : Occupancy
        decoder_input = decoder_input.unsqueeze(1)
        #print(decoder_input.shape)

        use_teacher_force = True if random.random() < teacher_force_ratio else False

        if use_teacher_force:
            # feed the target as next input
            for t in range(target_len):
                # Using precious hidden state which is context from encoder at start
                out, hidden = self.decoder(decoder_input, hidden)
                outputs[:, t] = out.squeeze(1)

                decoder_input = target[:, t]
                decoder_input = decoder_input.float().unsqueeze(1)
        else:
            # feed output as next input
            for t in range(target_len):
                out, hidden = self.decoder(decoder_input, hidden)
                outputs[:, t] = out.squeeze(1)

                output = out.clone()

                decoder_input = output.float()

        return outputs











