import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, in_size, out_size, hid_size, n_layers, dropout):
        super().__init__()

        self.linear_relu_stack = []
        for i in range(n_layers):
            if i == 0: layer_in = in_size
            else: layer_in = n_layers

            if i == n_layers-1: layer_out = out_size
            else: layer_out = n_layers

            self.linear_relu_stack.append(nn.Linear(layer_in, layer_out))
            if i != n_layers-1:
                self.linear_relu_stack.extend([nn.ReLU(), nn.Dropout(dropout)])

        self.linear_relu_stack = nn.Sequential(*self.linear_relu_stack)
    
    def forward(self, x):
        return self.linear_relu_stack(x)

