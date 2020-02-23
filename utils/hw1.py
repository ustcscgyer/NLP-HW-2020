from sklearn.utils import Bunch
import numpy as np

# Read data from file
def load_data(filename):
    """
    Load input data and return sklearn.utils.Bunch
    """
    target, text = [], []
    with open(filename, encoding="utf8") as file:
        for line in file:
            cols = line.split("\t")
            target.append(1 if cols[0] == "pos" else 0)
            text.append(cols[1].rstrip())

    return Bunch(text=text, target=np.array(target))


def save_prediction(arr, filename="prediction.csv"):
    """
    Save the prediction into file
    """
    out = open(filename, "w", encoding="utf8")
    for idx, val in enumerate(arr):
        pred = "pos" if val == 1 else "neg"
        out.write("%s,%s\n" % (idx, pred))
    out.close()
