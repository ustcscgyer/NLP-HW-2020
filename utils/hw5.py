from sklearn.utils import Bunch
import numpy as np
from keras import backend as K

# Read data from file
def load_data(filename):
    """
        load data from filename and return a list of lists
        data = [sent1, sent2, ...] where sent_i is
                a sequence of words 

        labels = [labs1, labs2, ...] where labs_i is a sequence of
                labels
    """
    data, labels = [], []
    sent, labs = [], []
    
    with open(filename, 'r') as file:
        file.readline() # Skip the license line
        for line in file:
            cols = line.rstrip().split("\t")
            if len(line.rstrip()) == 0: # sentence breaker
                if len(sent) == 0: continue
                data.append(sent)
                labels.append(labs)
                sent, labs = [], []
            else: 
                sent.append(cols[0])
                if len(cols) > 1: # training data
                    labs.append(cols[1].split("|")[0]) # In case of double label
        
        if len(sent) > 0:
            data.append(sent)
        if len(labs) > 0:
            labels.append(labs)

        return data, labels


def save_prediction(arr, filename="prediction.txt"):
    """
    Save the prediction into file
    """
    with open(filename, "w", encoding="utf8") as out:
        for labs in arr:
            for l in labs:
                out.write("%s\n" % l)
            out.write(" \n")


def check_path_search(path_search_method, examples):
    results = [path_search_method(example["Ms"]) == example["path"] 
               for example in examples]

    print("Success: %d / %d" % (sum(results), len(results)) )

 
def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)
 
        ignore_mask = K.cast(K.not_equal(y_true_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy
    return ignore_accuracy

def whole_sentence_accuracy(to_ignore=0):
    def sentence_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)
 
        ignore_mask = K.cast(K.equal(y_true_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') 
        matches = K.any(K.stack([matches, ignore_mask], axis=0), axis=0) # This is logic OR
        accuracy = K.sum(K.cast(K.all(matches, axis=1), 'int32')) / K.sum(K.cast(K.any(matches, axis=1), 'int32'))
                         
        return accuracy
    return sentence_accuracy
    