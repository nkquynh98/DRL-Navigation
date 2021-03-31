import torch 
import torch.nn as nn
import torch.nn.functional as F

class DQL_network(nn.Module):
    def __init__(self, state_values, action_values, seed, fc1_values = 64, fc2_values = 64):

        super(DQL_network, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_values, fc1_values)
        self.fc2 = nn.Linear(fc1_values, fc2_values)
        self.fc3 = nn.Linear(fc2_values, action_values)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.relu(self.fc3(x))
