# Read data from file
def load_data(filename='plot_summaries_tokenized.txt'):
    text = []
    with open(filename, encoding="utf8") as file:
        for line in file:
            tokens = line.strip().split(" ")
            text.append([t.lower() for t in tokens if t != ''])

    return text
