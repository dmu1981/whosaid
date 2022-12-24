#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
from dataset import generate_dataset, load_dataset
from model import WhoSaidModel, to_tensors

def predict(model, sentence):
  hidden = model.init()
  for index in range(len(sentence)):
    t = sentence[index].view(1,1)
    x, hidden = model.forward((t, hidden))

  return torch.argmax(x[0])

def play(model, datasets):
  print("Ok, let us play...")
  st = "validation"

  # Evaluation mode
  model.eval()
  
  player_score = 0
  computer_score = 0
  for iteration in range(10):
    # Pick a random sentence from the validation set
    index = int(random.uniform(0, len(datasets[st])))
    sentence = datasets[st][index][0]
    print("Who said: \"", end="")
    for word in sentence:
      print(datasets["vocab"][word] + " ", end="")
    print("\"?")
    while True:
      player_guess = input().upper()
      if player_guess in datasets["speakers"]:
        break
      
      print("Sorry, i don´t know this guy..")

    computer_guess = predict(model, sentence)
    computer_guess = datasets["speakers"][computer_guess]

    truth = datasets["speakers"][datasets[st][index][1]]
    print("I guess this was", computer_guess)
    print("In fact, it was", truth)

    if computer_guess == truth:
      computer_score += 1

    if player_guess == truth:
      player_score += 1
    
    print("Current score is:", computer_score," for me, ", player_score, " for you.")


print("Loading dataset...")
datasets = to_tensors(load_dataset("datasets.json"))

model = WhoSaidModel(len(datasets["vocab"]), len(datasets["speakers"]))

PATH="whosaid.pt"
try:
  checkpoint = torch.load(PATH)
  model.load_state_dict(checkpoint['model_state_dict'])
except:
  print("Could not load model... exit")
  exit()
  pass

play(model, datasets)
