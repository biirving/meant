import torch
from src.utils.torchUtils import weights_init
from torch import nn

class mlpEncoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden_layers=3, activation=nn.ReLU()):
        super().__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.num_hidden_layers=num_hidden_layers
        
        self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), activation)
        self.hidden = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                activation 
            ) for _ in range(num_hidden_layers)]
        )
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Sigmoid())
        self.apply(weights_init)

    def forward(self, **kwargs):
        x = kwargs.get('prices')
        x = self.input_layer(x)
        x = self.hidden(x)
        x = self.output_layer(x)
        return x


class LSTMEncoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden_layers=3, activation=nn.ReLU()):
        super().__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.num_hidden_layers=num_hidden_layers
        
        self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), activation)
        self.hidden = nn.LSTM(input_size = hidden_dim, hidden_size = hidden_dim, num_layers = num_hidden_layers)
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Sigmoid())
        self.apply(weights_init)

    def forward(self, **kwargs):
        x = kwargs.get('prices')
        x = self.input_layer(x)
        x = self.hidden(x)
        x = self.output_layer(x[0])
        return x

