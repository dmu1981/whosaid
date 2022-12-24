# Who Said 
## Introduction
"Who Said" is a recurrent deep-learning model which predicts who of the five Captains (Kirk, Picard, Sisko, Janyway, Archer) from the Star Trek Franchise said a given line of text. The model is trained on the publically available ["Star Trek Scripts"](https://www.kaggle.com/datasets/gjbroughton/start-trek-scripts) dataset on from [Kaggle](https://www.kaggle.com/).

## Raw data and pre-processing
Raw data is organized as a JSON document with each series and episode being represented individually as a giant string. The string itself contains the lines spoken by the characters as well as stage directions. An example of the raw-data looks like this

    \nSATIE: It must have been awful for you, actually becoming one of  them,\nbeing forced to use your vast knowledge of Starfleet   operations to aid\nthe Borg. Just how many of our ships were lost?  Thirty nine? And a loss\nof life, I believe, measured at nearly   eleven thousand. One wonders how\nyou can sleep at night, having  caused so much destruction. I question\nyour actions, Captain. I  question your choices. I question your\nloyalty. \nPICARD: You know, there some words I've known since I was a school boy.\nWith   the first link, the chain is forged. The first speech censured,\nthe   first thought forbidden, the first freedom denied, chains us all\nirrevocably. Those words were uttered by Judge Aaron Satie as wisdom\nand warning. The first time any man's freedom is trodden on, we're all\ndamaged. I fear that today

Note that this is a short snippet from Season 4, Episode 21 "The Drumhead" from Star Trek: The Next Generation. The whole script for that episode is much longer and the lines spoked by Picard are to be extracted from this before training can begin. The original dataset from Kaggle contains preprocessed lines per character, however the data is corrupted (words seperated by end-of-lines in the raw script were merged into a single word) and could not be used to train a good model. Thus, a dedicated pre-processing is included in this repository. 

The model is trained on sentences from five famous Captains from Star Trek, namely Captain James T. Kirk (The original series), Benjamin Sisko (Deep Space 9), Kathryn Janeway (Voyager), Jean Luc Picard (The next generation) and Jonathan Archer (Enterprise).
All words used by any of these characters are extracted. Then, a vocabulary of words is being created by selecting the 95% most frequently used words over all five characters. The vocabulary contains 4388 words. Sentences containing words not represented in the vocabulary are replaced with an "UNKNOWN" symbol. The data is then split into a balanced training set (same amount of sentences for each character). Lines not used for training are put into the (unbalanced) validation set. 

A single sample is represented as a tupel, containing a list of symbols representing the spoken sentence as well as the ground truth label for who has said this line, for example

    (89, 193, 22, 2398, 2, 20, 15, 4389], "KIRK")

The indices can be translated back into words using the vocabulary. 
Note: Sentences have varying length, so have the respective samples. 

You can run

    python preprocess.py

to preprocess the data. You can edit the code to select other characters or make the task more difficult by adding additional speakers to the dataset. 

## The model
The model is trained using PyTorch. A simple architecture is chosen where embeddings for each symbol (word) are learned. A 256-dimensional vector space is arbitrarily chosen to represent the embeddings. The embeddings are feed into a three-layer GRU (Gated Recurrent Unit) with a 512-dimensional hidden state. The output of the last unit is mapped fully-connected to the 5-dimensional output representing the likelihood of each character having said the given input sentence. 

The model is trained with drop-out (p=0.5) in each GRU layer, minimal weight-decay and an exponentially decreasing learning rate schedule. 

The model achieves an accuracy of ~45% on the validation set. Note that, while this may sound little, the scripts contain sentences like 

    JANEWAY: Pardon me?
    KIRK: Who started the fight?
    SISKO: Full reverse.
    PICARD: Have we met before?
    ARCHER: What do you like?

which, to be realistic, could have been said by all of the respective Captains. So, the problem at hand is really hard, thus 45% represents a fair performance (note that random guesssing would mean 20% accuracy). 

Run

    python train.py

to train the model. The script will continue training from the last saved checkpoint, but it will always start the learning-rate schedule from scratch. Delete the "whosaid.pt" file or comment the loading code to start from a randomly initialized network.  

## The Guessing Game
Wanna challenge the model and see if you can do better? Run

    python game.py

to play against the model. A random choice of 10 sentences is selected from the validation set and you are asked to answer which of the five Captains has said this sentence. The model is doing is own prediction and a score is counted to see, who was correct more often. Can you beat the model?

## About the Author
I am a passionate developer with a professional AI/ML background. I wanted to experiment with natural language processing and recurrent neural networks, so i came up with the idea for this little experiment. Use the code with great care as there are probably ways to do the same more efficiently or achieve better performance. 