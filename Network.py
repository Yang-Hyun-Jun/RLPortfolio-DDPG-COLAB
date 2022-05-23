import torch
import torch.nn as nn

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

class Actor(nn.Module):
    def __init__(self, K, state1_dim=5, state2_dim=2):
        super().__init__()
        self.K = K
        self.layer_s = nn.Linear(state1_dim + state2_dim, 64)
        self.layer1 = nn.Linear(64*K, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 3)
        self.hidden_act = nn.ReLU()

        nn.init.kaiming_normal_(self.layer1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.layer2.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.layer3.weight, nonlinearity="relu")

    def forward(self, s1_tensor, portfolio):
        """
        state = (s1_tensor, portfolio)
        s1_tensor: (batch, assets, features)
        """

        for i in range(s1_tensor.shape[1]):
            s = torch.cat([s1_tensor[:,i,:], portfolio[:,0], portfolio[:,i+1]], dim=-1)
            globals()[f"score{i+1}"] = self.layer_s(s)

        for j in range(s1_tensor.shape[1]):
            s_vec = list() if j == 0 else s_vec
            s_vec.append(globals()[f"score{j+1}"])

        x = torch.cat(s_vec, dim=-1)
        x = self.layer1(x)
        x = self.hidden_act(x)
        x = self.layer2(x)
        x = self.hidden_act(x)
        x = self.layer3(x)

        cash_bias = torch.ones(size=(x.shape[0],1), device=device) * 0.1
        x = torch.cat([cash_bias, x], dim=-1)
        portfolio = torch.softmax(x, dim=-1)
        return portfolio


class Qnet(nn.Module):
    def __init__(self, K, state1_dim=5, state2_dim=2):
        super(Qnet, self).__init__()
        self.K = K

        # header
        self.layer_s = nn.Linear(state1_dim + state2_dim, 64)
        self.layer_a = nn.Linear(K+1, 64)
        self.layer1 = nn.Linear(64*(K+1), 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)
        self.hidden_act = nn.ReLU()

        nn.init.kaiming_normal_(self.layer1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.layer2.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.layer3.weight, nonlinearity="relu")

    def forward(self, s1_tensor, portfolio, action):

        for i in range(s1_tensor.shape[1]):
            s = torch.cat([s1_tensor[:,i,:], portfolio[:,0], portfolio[:,i+1]], dim=-1)
            globals()[f"score{i+1}"] = self.layer_s(s)

        for j in range(s1_tensor.shape[1]):
            s_vec = list() if j == 0 else s_vec
            s_vec.append(globals()[f"score{j+1}"])

        a_vec = self.layer_a(action)
        x = torch.cat(s_vec + [a_vec], dim=-1)
        x = self.layer1(x)
        x = self.hidden_act(x)
        x = self.layer2(x)
        x = self.hidden_act(x)
        x = self.layer3(x)
        return x



if __name__ == "__main__":
    s1_tensor = torch.rand(size=(10, 3, 5))
    portfolio = torch.rand(size=(10, 4, 1))
    action = torch.rand(size=(10, 3))

    actor = Actor(K=3)
    qnet = Qnet(K=3)
    print(actor(s1_tensor,portfolio).shape)
    print(qnet(s1_tensor,portfolio,action).shape)