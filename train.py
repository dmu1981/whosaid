#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
from dataset import generate_dataset, load_dataset
from model import WhoSaidModel, to_tensors

device = "cuda" if torch.cuda.is_available() else "cpu"
PATH = "whosaid.pt"

def epoch(model, dataset, training):
  # Shuffle dataset  
  random.shuffle(dataset)

  # Enable train mode
  if training:
    name = 'Train'
    model.train()
  else:
    name = 'Val'
    model.eval()

  cnt = 0
  total_loss = 0
  correct = 0
  
  for sentence, label in dataset:
    model.zero_grad()
    hidden = model.init()
    loss = 0
    for index in range(len(sentence)):
      t = sentence[index].view(1,1)
      x, hidden = model.forward((t, hidden))
      loss += loss_function(x[0], label)
    loss /= len(sentence)

    if torch.argmax(x[0]) == label[0]:
      correct += 1    
    
    if training:
      loss.backward()
      optimizer.step()

    total_loss += loss.item()
    cnt = cnt + 1
    if cnt % 500 == 0:
      print(name,': ', cnt, "/", len(dataset),": loss ", total_loss / cnt, "    accuracy", "{:.2f}%".format(correct / cnt * 100))

  print("   Final accuracy", "{:.2f}%".format(correct / cnt * 100))
  return correct / cnt      

print("Loading dataset...")
datasets = to_tensors(load_dataset("datasets.json"))

model = WhoSaidModel(len(datasets["vocab"]), len(datasets["speakers"]))
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

try:
  checkpoint = torch.load(PATH)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  model.train()
except:
  print("Could not load model, starting from scratch")
  pass


loss_function = nn.NLLLoss()
best_accuracy = 0
for i in range(10):
  print("Epoch ", i)
  epoch(model, datasets["train"], True)
  accuracy = epoch(model, datasets["validation"], False)
  scheduler.step()
  if accuracy > best_accuracy:
    best_accuracy = accuracy
    print("This is the best model so far, saving it")
    torch.save({
              'epoch': i,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              }, PATH)

          
