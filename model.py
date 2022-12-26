"""Definition of the WhoSaid model"""
import torch
from torch import nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBEDDING_DIM = 256

def to_tensors(datasets):
    """Convert dataset indices to torch tensors and move them to target device"""
    train = []
    val = []
    for sample in datasets["train"]:
        index = torch.tensor(datasets["speakers"].index(sample[1]),dtype=torch.long).view(1)
        tensors = [torch.tensor(sample[0]).to(DEVICE), index.to(DEVICE)]
        train.append(tensors)

    for sample in datasets["validation"]:
        index = torch.tensor(datasets["speakers"].index(sample[1]),dtype=torch.long).view(1)
        tensors = [torch.tensor(sample[0]).to(DEVICE), index.to(DEVICE)]
        val.append(tensors)

    datasets["train"] = train
    datasets["validation"] = val

    return datasets

class WhoSaidModel(nn.Module):
    """The WhoSaid Model"""
    def __init__(self, vocab_size, nr_speakers):
        super(WhoSaidModel, self).__init__()
        self.hidden_size = EMBEDDING_DIM * 2
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM).to(DEVICE)
        self.gru = nn.GRU(input_size = EMBEDDING_DIM,
                          num_layers=3,
                          hidden_size = self.hidden_size,
                          dropout=0.5).to(DEVICE)
        self.linear = nn.Linear(self.hidden_size, nr_speakers).to(DEVICE)
        self.relu = nn.ReLU()

    def forward(self, token):
        """forward pass"""
        state, hidden = token
        state = self.embedding(state)
        state, hidden = self.gru(state, hidden)
        state = self.linear(state)
        state = F.log_softmax(state, dim=2)
        return state, hidden

    def init(self):
        """weight initialization"""
        weight = next(self.parameters()).data
        hidden = weight.new(3, 1, self.hidden_size).zero_()
        return hidden
