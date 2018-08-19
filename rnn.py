import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.emb = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        #self.softmax = nn.Softmax()

    def forward(self, x):
        embedded = self.emb(x)
        output, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))
