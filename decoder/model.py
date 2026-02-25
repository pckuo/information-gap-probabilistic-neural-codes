# for submission to ICLR 2026


import torch.nn as nn
import torch.nn.functional as F


class NNDecoder(nn.Module):
    '''
    hidden_dim = (hidden1, hidden2): hidden dimensions for the two layers
    dropout_rate = (dropout_rate1, dropout_rate2): dropout rate for the two layers
    '''
    
    def __init__(
        self,
        input_dim,
        layers,
        dropout_rates,
        output_dim
    ):
        super(NNDecoder, self).__init__()


        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = layers
        self.dropout_rates = dropout_rates

        # initialize layers
        curr_input_dim = input_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(self.layers)):
            fc = nn.Linear(curr_input_dim, self.layers[i])
            self.fc_layers.append(fc)
            curr_input_dim = layers[i]
        self.output_layer = nn.Linear(curr_input_dim, self.output_dim)

        
    def forward(self, x):
        for i in range(len(self.layers)):
            dropout = nn.Dropout(self.dropout_rates[i])
            x = F.leaky_relu(dropout(self.fc_layers[i](x)))
        output = self.output_layer(x)

        return output