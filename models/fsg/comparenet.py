import torch


class CompareNet(torch.nn.Module):
    def __init__(self, input_dim=200, map1_dim=2048, map2_dim=64):
        super().__init__()

        self.fc1 = torch.nn.Linear(input_dim, map1_dim)
        self.relu_fc1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(map1_dim * 3, map2_dim)
        self.relu_fc2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(map2_dim, 2)

    def forward(self, x):
        x1, x2 = x
        m1_x1 = self.relu_fc1(self.fc1(x1))
        m1_x2 = self.relu_fc1(self.fc1(x2))

        x1x2_mult = m1_x1 * m1_x2
        x1x2_concat = torch.concat([m1_x1, x1x2_mult, m1_x2], dim=1)

        m2 = self.relu_fc2(self.fc2(x1x2_concat))
        out = self.fc3(m2)

        return out
