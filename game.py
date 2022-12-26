"""The WhoSaid Game"""
import random
import sys
import torch
from dataset import load_dataset
from model import WhoSaidModel, to_tensors

def predict(model, sentence):
    """Predict a sentence with the WhoSaid model"""
    hidden = model.init()
    for index, _ in enumerate(sentence):
        txt = sentence[index].view(1,1)
        state, hidden = model.forward((txt, hidden))

    return torch.argmax(state[0])

def play(model, datasets):
    """Play a game"""
    print("Ok, let us play...")
    dataset = "validation"

    # Evaluation mode
    model.eval()

    player_score = 0
    computer_score = 0
    for _ in range(10):
        # Pick a random sentence from the validation set
        index = int(random.uniform(0, len(datasets[dataset])))
        sentence = datasets[dataset][index][0]
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

        truth = datasets["speakers"][datasets[dataset][index][1]]
        print("I guess this was", computer_guess)
        print("In fact, it was", truth)

        if computer_guess == truth:
            computer_score += 1

        if player_guess == truth:
            player_score += 1

        print("Current score is:", computer_score," for me, ", player_score, " for you.")


print("Loading dataset...")
dset = to_tensors(load_dataset("datasets.json"))

mdl = WhoSaidModel(len(dset["vocab"]), len(dset["speakers"]))

PATH="whosaid.pt"
try:
    checkpoint = torch.load(PATH)
    mdl.load_state_dict(checkpoint['model_state_dict'])
except:
    print("Could not load model... exit")
    sys.exit()

play(mdl, dset)
