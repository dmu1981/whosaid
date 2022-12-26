"""Data preprocessing for the WhoSaid model"""
import json
import re
from collections import Counter
import random

def split_line(line):
    """Split a line into words, removing non-alphabetic symbols and make everything lowercase"""
    txt = line.split()
    txt = [''.join(filter(str.isalpha, x)).lower() for x in txt]
    return txt

def get_character(line):
    """Extract the character speaking on this line, return None is none is found"""
    txt = line.split(':')
    if len(txt) !=2:
        return None, line.strip()

    if txt[0].isupper():
        return txt[0].strip(), txt[1].strip()

    return None, line.strip()

def parse_raw(filename):
    """Parse the raw scripts and extract character lines"""
    # Open the file
    with open(filename,"r",encoding='utf-8') as file:
        # Parse JSON
        corpus = json.loads(file.read())

    character_lines = {}

    # Iterate all series
    for series in corpus.keys():
        print(series)
        # Iterate all episodes in this series
        for episode in corpus[series].keys():
            # Get the raw script
            script = corpus[series][episode]
            # Remove stuff in brackets (Regieanweisungen)
            script = "".join(re.split("\(|\)|\[|\]", script)[::2])
            # Split at new-lines
            script = script.split('\n')
            # Iterate over all lines,
            current_character = None
            current_line = ""
            for line in script:
                # Identify character speaking
                char, restline = get_character(line)

                # Some scripts end with part of the original HTML... skip that
                if restline.startswith("<Back"):
                    break

                # If there is no character change of the character is
                # continuing to speak, just add to this line
                if char is None or char == current_character:
                    current_line += " " + restline
                else:
                    # If there is a new character speaking
                    if char != current_character:
                        # And there was an actual character speaking
                        if current_character is not None:
                            # Remove duplicated whitespaces
                            current_line = re.sub(' +', ' ', current_line).strip()
                            # Add to our list
                            if current_character in character_lines:
                                character_lines[current_character] += [current_line]
                            else:
                                character_lines[current_character] = [current_line]

                        # Remember this line as the new current line
                        current_line = restline

                        # And remember the current character speaking
                        current_character = char

            current_line = re.sub(' +', ' ', current_line).strip()
            if current_character in character_lines:
                character_lines[current_character] += [current_line]
            else:
                character_lines[current_character] = [current_line]

    return character_lines

def build_vocabulary(speakers, lines):
    """Build the vocabulary for the relevant speakers by finding a subset of
       words representing 95% of all spoken words.
    """
    # Create a list of all words from the speakers
    words = []
    character_words = {}
    for speaker in speakers:
        character_words[speaker] = []

    # Create list of all words and list of words per character
    for speaker in speakers:
        for line in lines[speaker]:
            splt = split_line(line)
            words += splt
            character_words[speaker] += splt

    # Create counter
    counter_all = Counter(words)
    character_counter = {}
    for speaker in speakers:
        character_counter[speaker] = Counter(character_words[speaker])

    # Create unique vocabulary
    vocab = list(set(words))
    print("Vocabulary contains", len(vocab), "unique words out of", len(words), "total words")

    # Collect 95% quantil of unique words to represent the character lines best
    quantil = len(words) * 19 / 20
    sorted_words = sorted(counter_all.items(), key=lambda x: x[1], reverse=True)
    word_subset = []
    while quantil > 0:
        wrd = sorted_words.pop(0)
        word_subset.append(wrd[0])
        quantil -= wrd[1]

    print("95% quantil contains", len(word_subset), "unique words")

    # Append start, stop and unknown symbol for later usage
    word_subset.append("<STOP>")
    word_subset.append("<UNKNOWN>")
    return word_subset



def translate(speakers, lines, vocab):
    """Translate words into their respective symbols using the provided vocabulary.
        Replace words not represented in the vocublary with an UNKNOWN symbol.
        Append a STOP symbol at each sentence.
    """
    STOP_SYMBOL = vocab.index("<STOP>")
    UNKNOWN_SYMBOL = vocab.index("<UNKNOWN>")

    def index_or_unknown(vocab, wrd):
        if wrd in vocab:
            return vocab.index(wrd)

        return UNKNOWN_SYMBOL

    symbols = {}
    for speaker in speakers:
        symbols[speaker] = []

    for speaker in speakers:
        for line in lines[speaker]:
            splt = split_line(line)
            splt = [index_or_unknown(vocab, x) for x in splt] + [STOP_SYMBOL]
            symbols[speaker].append(splt)

    return symbols

def split_sets(vocab, speakers, symbols):
    """Split the data into training and validation sets"""
    train_set_per_speaker = 9999999
    for speaker in symbols.keys():
        if len(symbols[speaker]) < train_set_per_speaker:
            train_set_per_speaker = len(symbols[speaker])

    train_set_per_speaker = int(train_set_per_speaker * 0.8)
    print("Training set is",train_set_per_speaker,"lines per character")

    train_set = []
    val_set = []
    
    for speaker in symbols.keys():
        # Pick random subsets
        random.shuffle(symbols[speaker])
        train_set += [(x, speaker) for x in symbols[speaker][0:train_set_per_speaker]]
        val_set += [(x, speaker) for x in symbols[speaker][train_set_per_speaker:-1]]

    # Shuffle both sets
    sets = {
        "vocab": vocab,
        "speakers": speakers,
        "train": train_set,
        "validation": val_set,
      }

    return sets

def load_dataset(filename):
    """Load and parse a previously stored dataset file"""
    with open(filename,"r",encoding='utf-8' ) as file:
        sets = json.loads(file.read())
        file.close()
    return sets

def generate_dataset(filename, sets_fname):
    """Generate a new dataset file for a given list of speakers"""
    captains = ['KIRK', 'PICARD', 'SISKO', 'JANEWAY', 'ARCHER']

    character_lines = parse_raw(filename)
    vocab = build_vocabulary(captains, character_lines)
    symbols = translate(captains, character_lines, vocab)
    sets = split_sets(vocab, captains, symbols)

    with open(sets_fname,"w",encoding='utf-8') as file:
        file.write(json.dumps(sets))
        file.close()
