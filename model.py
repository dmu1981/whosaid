import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

EMBEDDING_DIM = 256

def to_tensors(datasets):
  train = []
  val = []
  n_speakers = len(datasets["speakers"])
  for sample in datasets["train"]:
    index = torch.tensor(datasets["speakers"].index(sample[1]),dtype=torch.long).view(1)
    tensors = [torch.tensor(sample[0]).to(device), index.to(device)]
    train.append(tensors)

  for sample in datasets["validation"]:
    index = torch.tensor(datasets["speakers"].index(sample[1]),dtype=torch.long).view(1)
    tensors = [torch.tensor(sample[0]).to(device), index.to(device)]
    val.append(tensors)

  datasets["train"] = train
  datasets["validation"] = val

  return datasets
  
class WhoSaidModel(nn.Module):
  def __init__(self, vocab_size, nr_speakers):
    super(WhoSaidModel, self).__init__()
    self.hidden_size = EMBEDDING_DIM * 2
    self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM).to(device)
    self.gru = nn.GRU(input_size = EMBEDDING_DIM, num_layers=3, hidden_size = self.hidden_size, dropout=0.5).to(device)
    self.linear = nn.Linear(self.hidden_size, nr_speakers).to(device)
    self.relu = nn.ReLU()

  def forward(self, token):
    x, h = token
    x = self.embedding(x)
    x, h = self.gru(x, h)
    x = self.linear(x)
    x = F.log_softmax(x, dim=2)
    return x, h
  
  def init(self):
    weight = next(self.parameters()).data
    hidden = weight.new(3, 1, self.hidden_size).zero_()
    return hidden