import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, n_in, n_out):
        super(DQN, self).__init__()

        self.ln1 = nn.Linear(n_in, 32)
        self.ln2 = nn.Linear(32, 32)
        # self.ln3 = nn.Linear(32, 32)
        self.ln4 = nn.Linear(32, 32)
        self.out = nn.Linear(32, n_out)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        # x = F.relu(self.ln3(x))
        x = F.relu(self.ln4(x))
        return self.out(x)

