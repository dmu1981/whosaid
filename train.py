#-*- coding:utf-8 -*-
"""Training code for the WhoSaid model"""
import random
import torch
from torch import nn
from torch import optim
from dataset import load_dataset
from model import WhoSaidModel, to_tensors

#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATH = "whosaid.pt"

def epoch(model, dataset, training):
    """Do one epoch on model given the dataset. If training is True, also do backpropagation"""
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
        for index, _ in enumerate(sentence):
            symbol = sentence[index].view(1,1)
            state, hidden = model.forward((symbol, hidden))
            loss += loss_function(state[0], label)
        loss /= len(sentence)

        if torch.argmax(state[0]) == label[0]:
            correct += 1

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        cnt = cnt + 1
        if cnt % 500 == 0:
            print(name, ': ',
                  cnt, "/", len(dataset),
                  ": loss ", total_loss / cnt,
                  "    accuracy", "{:.2f}%".format(correct / cnt * 100))

    print("   Final accuracy", "{:.2f}%".format(correct / cnt * 100))
    return correct / cnt

print("Loading dataset...")
datasets = to_tensors(load_dataset("datasets.json"))

mdl = WhoSaidModel(len(datasets["vocab"]), len(datasets["speakers"]))
optimizer = optim.Adam(mdl.parameters(), lr=0.0001, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

try:
    checkpoint = torch.load(PATH)
    mdl.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    mdl.train()
except:
    print("Could not load model, starting from scratch")


loss_function = nn.NLLLoss()
best_accuracy = 0
for i in range(10):
    print("Epoch ", i)
    epoch(mdl, datasets["train"], True)
    accuracy = epoch(mdl, datasets["validation"], False)
    scheduler.step()
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        print("This is the best model so far, saving it")
        torch.save({
                  'epoch': i,
                  'model_state_dict': mdl.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  }, PATH)
