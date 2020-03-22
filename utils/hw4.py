import re, string
from collections import Counter
import numpy as np
import pandas as pd

START = "<START>"
UNK = "<UNK>"
END = "<END>"

def load_data(filename):
    """
    Load the training data into the training format
    remove punctuation and return a list of tokens
    """
    with open(filename, 'r') as f:
        headlines = f.readlines()
        
    # Removing excess punctuation and newline
    pattern = re.compile('[%s]' % re.escape(string.punctuation))
    headlines = [pattern.sub('', h.strip("\n")).split(' ') for h in headlines]

    return headlines

def load_data_char(filename):
    """
    Load the training data into the training format
    Return a list of characters
    """
    with open(filename, 'r') as f:
        headlines = f.readlines()
    
    headlines = [list(h.strip('\n')) for h in headlines]
    
    return headlines

def gen_vocab(dataset, min_token_ct=0):
    """
    For given training data, list of vocabulary list, i.g.
    [["this", "set", "1"],
     ["this", "is", "another", "set"],
     ]
     
    return the vocab list and rev_vocab dictionary
    3 numerical encodings are reserved: {<UNK>:0, <START>:1, <END>:2}
    """
    token_ct = Counter([token for row in dataset for token in row])
    token_ct = {k: v for k, v in token_ct.items() if v >= min_token_ct}
    vocab = sorted(token_ct, key=token_ct.get, reverse=True)
    vocab = [UNK, START, END] + vocab
    
    rev_vocab = {fea: fid for fid, fea in enumerate(vocab)}
    
    return vocab, rev_vocab

def load_embedding(filename, vocab=None):
    """
    Load the embedding file into a pandas DF
    
    If a vocab set is provided, only return the subset in the vocab list, if tokens
    in the vocab list is not present in the embedding, use randomalized value
    """
    embedding = pd.read_csv(filename)
    if vocab:
        m = []
        normalize = (embedding**2).sum(axis=0).mean()
        embedding_dim = embedding.shape[0]
        
        for t in vocab:
            v = embedding.get(t)
            if v is None:
                v0 = np.random.rand(embedding.shape[0]) - 0.5
                # apply normalization so the expected module is equal to the 
                # average module of the embedding matrix
                v = v0 * 2 * np.random.rand() * np.sqrt(normalize / (v0**2).sum())
                
            m.append(v)
            
        embedding = pd.DataFrame(m , index=vocab)
    
    return embedding

    

