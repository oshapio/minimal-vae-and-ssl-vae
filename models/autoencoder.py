import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, input_size=2, hidden_size=10, bottleneck_size=1):
        super(AutoEncoder, self).__init__()
        self.input_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_bottleneck = nn.Linear(hidden_size, bottleneck_size)
        self.bottleneck_hidden = nn.Linear(bottleneck_size, hidden_size)
        self.hidden_output = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = F.relu(self.input_hidden(x))
        encoded = self.hidden_bottleneck(x)
        x = F.relu(self.bottleneck_hidden(encoded))
        decoded = self.hidden_output(x)
        return {"encoded": encoded, "decoded": decoded}