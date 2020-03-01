from .hw1 import load_data, save_prediction

# Read the vocabulary file
def read_vocab(filename="vocab.txt"):
    """
    Load the vocab file in to the vocabulary dictionary
    """
    vocab = {}
    with open(filename, 'r') as file:
        for line in file:
            cols = line.rstrip().split("\t")
            vocab[cols[0]] = int(cols[1])

    return vocab
