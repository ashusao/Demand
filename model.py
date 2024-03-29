import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from data import Data
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
#torch.set_deterministic(True) # type: ignore

class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, dropout, num_layers=1):
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
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input, hidden):
        '''
        :param input:       input of shape (batch, seq_len, input_size)
        :param hidden:      initial hidden state (num_layers, batch_size, hidden_size)
        :return: output, hidden
                output gives all outputs in seq. shape (batch_size, seq_len, output_size)
                hidden represents context vector. shape (num_layers, batch_size, hidden_size)
        '''
        output, hidden = self.gru(input, hidden)
        output = self.dropout(output)
        return output, hidden

    def init_hidden(self, batch_size):
        '''
        :param batch_size:      input.shape[0]
        :return:                zeroed initialized hidden states
        '''
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, dropout, num_layers=1):
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
        #self.feat_size_cs = feat_size_cs
        #self.feat_size_spatial = feat_size_spatial
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        #self.linear1 = nn.Linear(hidden_size + feat_size_cs + feat_size_spatial, hidden_size)
        #self.linear3 = nn.Linear(256, hidden_size)
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
        output = self.dropout(output)

        #if decode == 'features':
        #    output = torch.cat((output, features.unsqueeze(1)), 2) # concat decoder output and features
        #    output = F.relu(self.linear1(output))

        #output = F.relu(self.linear2(output))
        #output = F.relu(self.linear3(output))
        out = self.linear(output)

        # squeeze the seq_len dimension so that output is (batch_size, output_dim)
        out = out.squeeze(1)
        out = torch.sigmoid(out)
        out = torch.clamp(out, 0, 1)
        return out, hidden

class Embedding(nn.Module):

    def __init__(self, feat_size, embed_size):
        super(Embedding, self).__init__()
        self.feat_size = feat_size
        self.embed_size = embed_size
        self.linear = nn.Linear(feat_size, embed_size)

    def forward(self, features):
        return F.relu(self.linear(features))


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, embedding_cs, embedding_spatial, embedding_pattern, embedding_median,
                 embedding_q25, embedding_q75, embedding, config):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        feat = self.config.getboolean('data', 'features')
        if feat and self.config['model']['decoder'] == 'features':
            #self.embedding_cs = embedding_cs
            #self.embedding_spatial = embedding_spatial
            self.embedding_pattern = embedding_pattern
            #self.embedding_median = embedding_median
            self.embedding_q25 = embedding_q25
            self.embedding_q75 = embedding_q75
            self.embedding = embedding
        self.data_obj = Data()

    def forward(self, source, target, features_cs, features_spatial, features_pattern, features_median,
                features_q25, features_q75, teacher_force_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]

        #output_size = target.shape[2]

        #outputs = torch.zeros(batch_size, target_len, output_size).to(device)
        outputs = torch.zeros(batch_size, target_len).to(device)

        feat = self.config.getboolean('data', 'features')
        decode = self.config['model']['decoder']
        num_layers = int(self.config['train']['num_layers'])

        hidden = self.encoder.init_hidden(batch_size).to(device)
        encoder_out, hidden = self.encoder(source, hidden)

        if feat and decode == 'features':
            #features_cs = features_cs.unsqueeze(0)  # add extra dimensino for num_layers
            #features_spatial = features_spatial.unsqueeze(0)
            #features_median = features_median.unsqueeze(0)
            features_pattern = features_pattern.unsqueeze(0)
            features_q25 = features_q25.unsqueeze(0)
            features_q75 = features_q75.unsqueeze(0)

            #features_cs = features_cs.repeat(hidden.shape[0], 1, 1)  # copy features to each layers (num_layers, batch, hidden_size)
            #features_spatial = features_spatial.repeat(hidden.shape[0], 1, 1)
            #features_median = features_median.repeat(hidden.shape[0], 1, 1)
            features_pattern = features_pattern.repeat(num_layers, 1, 1)
            features_q25 = features_q25.repeat(num_layers, 1, 1)
            features_q75 = features_q75.repeat(num_layers, 1, 1)

            #features_cs = self.embedding_cs(features_cs)
            #features_spatial = self.embedding_spatial(features_spatial)
            #features_median = self.embedding_median(features_median)
            features_pattern = self.embedding_pattern(features_pattern)
            features_q25 = self.embedding_q25(features_q25)
            features_q75 = self.embedding_q75(features_q75)

            concat = torch.cat((hidden, features_pattern, features_q25, features_q75), 2)  # (num_layers, batch, hidden_size + feat_size)

            hidden = self.embedding(concat)

        # input the state of charger without features
        decoder_input = source[:, -1, 0]  # [0] : Occupancy             #here
        decoder_input = decoder_input.unsqueeze(1)

        use_teacher_force = True if random.random() < teacher_force_ratio else False

        if use_teacher_force:
            # feed the target as next input
            for t in range(target_len):
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
                #output = torch.cat((output.float(), target[:, t, 1:]), 1)  # here
                decoder_input = output.float()

        return outputs











