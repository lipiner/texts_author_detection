from os import path, listdir
from collections import Counter
import pickle

DATA_DIR = "Gutenberg/txt"
MIN_TEXTS_AMNT = 5


def extract_data():
    """
    Extract the texts and the labels from the database
    :return: list of tuples (raw_text, label). Raw text is the string contains the full text. label is the text's
    label (author)
    """
    data_files = listdir(DATA_DIR)
    data = list()
    labels = list()

    for file in data_files:
        label = file.split("___")[0]
        with open(path.join(DATA_DIR, file), 'r', encoding="ISO-8859-1") as f:
            text = f.read()
        data.append((text, label))
        labels.append(label)

    # finds authors with more than the minimun texts amount per authors
    authors = get_values_above_t(Counter(labels), MIN_TEXTS_AMNT)

    # filter out authors with less than 5 textx
    filtered_data = list()
    for text, label in data:
        if label in authors:
            # filtered_data.append((text, authors.index(label)))
            filtered_data.append((text, label))

    # with open("authors_labels.pickle", 'wb') as f:
    #     pickle.dump(authors, f)
    with open("data.pickle", 'wb') as f:
        pickle.dump(filtered_data, f)

    return filtered_data


def get_values_above_t(data, threshold):
    """
    Retruns a list with all the keys of the given data that their values is a above the threshold
    :param data: a dictionary form
    :param threshold: the minimum value for the keys
    :return: list with all the keys of the given data that their values is a above the threshold
    """
    keys = list()
    for key, value in data.items():
        if value >= threshold:
            keys.append(key)
    return keys

extract_data()
