import numpy as np
import nltk
import pickle
from collections import Counter
import string
from preprocessing import get_values_above_t
import spacy

BIGRAMS_THRESHOLD = 50

# downloads nltk data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
dep_parser = spacy.load('en')


def get_function_words():
    """
    :return: list with function words
    """
    with open('function_words.txt') as f:
        function_words = [line.rstrip('\n') for line in f]
    return function_words


def tokenize_text(text):
    """
    Tokenize the given text into list of tokens without punctuation tokens
    :param text: string
    :return: list of tokens without punctuation tokens
    """
    tokens = nltk.word_tokenize(text)
    tokens_without_punc = [token for token in tokens if token not in string.punctuation]
    return tokens_without_punc


def get_function_words_features(texts):
    """
    Creates features matrix of function words frequency.
    :param texts: list of tuples (raw_text, label). Raw text is the string contains the full text. label is the text's
    label (author)
    :return: a tuple of matrix features and labels vector
    """
    function_words = get_function_words()

    feature_matrix = np.zeros((len(texts), len(function_words)))
    label_vector = np.empty(len(texts), dtype='object')
    for i, (text, author) in enumerate(texts):
        feature_matrix[i] = calc_fwords_freqs(text, function_words)
        label_vector[i] = author

    return feature_matrix, label_vector


def calc_fwords_freqs(text, function_words):
    """
    Calculates the frequency of each word from function_words in the given text
    :param text: the text to search in
    :param function_words: a list of function words that their frequency in the text will be calculated
    :return: an array which its i'th entry contains frequency of word i in function_words
    """
    tokens = tokenize_text(text)
    cnt = Counter(tokens)
    word_num = len(cnt)
    features = np.zeros(len(function_words))
    for i, fword in enumerate(function_words):
        features[i] = cnt[fword] / word_num

    return features


def extract_POS_bigrams(text):
    """
    Finds all the part of speech bigrams in the given text
    :param text: string
    :return: a dictionary with all the POS bigrams as keys and their amount in the text as values
    """
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    bigarms = list()
    # creates bigrams list of the text
    for i in range(len(pos_tags) - 1):
        bigarms.append((pos_tags[i][1], pos_tags[i+1][1]))

    return Counter(bigarms)  # convert to dictionary with each bigram amount


def extract_dependency_bigrams(text):
    """
    Finds all the dependency bigrams in the given text
    :param text: string
    :return: a dictionary with all the dependency bigrams as keys and their amount in the text as values
    """
    parsed_text = dep_parser(text)
    dependency_tags = [word.dep_ for word in parsed_text]
    bigarms = list()
    # creates bigrams list of the text
    for i in range(len(dependency_tags) - 1):
        bigarms.append((dependency_tags[i], dependency_tags[i+1]))

    return Counter(bigarms)  # convert to dictionary with each bigram amount


def get_POS_data(texts):
    """
    Creates features matrix of part of speech bigrams
    :param texts: list of tuples (raw_text, label). Raw text is the string contains the full text. label is the text's
    label (author)
    :return: a tuple of matrix features and labels vector
    """
    return get_bigrams_features(texts, extract_POS_bigrams)


def get_dependency_data(texts):
    """
    Creates features matrix of dependency bigrams
    :param texts: list of tuples (raw_text, label). Raw text is the string contains the full text. label is the text's
    label (author)
    :return: a tuple of matrix features and labels vector
    """
    return get_bigrams_features(texts, extract_dependency_bigrams)


def get_bigrams_features(texts, bigrams_func):
    """
    Creates features matrix of bigrams that appears more than a threshold, with the given bigrams function.
    :param texts: list of tuples of (text, author)
    :param bigrams_func: function that gets texts (list with tuples of text and its label), extract bigrams
    and returns a dictionary of the bigrams as keys and their amount in the text as values
    :return: a tuple of matrix features and labels vector
    """
    labels = np.empty(len(texts), dtype='object')
    texts_bigrams = np.empty(len(texts), dtype='object')
    bigrams_amnts = Counter()
    # gets the bigrams in each text
    for i, text in enumerate(texts):
        labels[i] = text[1]

        bigrams = Counter()
        for sent in nltk.tokenize.sent_tokenize(text[0]):
            bigrams += bigrams_func(sent)
        texts_bigrams[i] = bigrams
        bigrams_amnts += bigrams  # keeps track of all bigrams

    # creates matrix features for each text according to the most common bigrams
    final_bigrams = get_values_above_t(bigrams_amnts, BIGRAMS_THRESHOLD)
    features = np.zeros((len(texts), len(final_bigrams)))
    for i, text in enumerate(texts_bigrams):
        for j, bigrams in enumerate(final_bigrams):
            value = text.get(bigrams)
            if value is not None:
                features[i][j] = value / len(texts[i])

    with open("%s_features_names.pickle" % bigrams_func.__name__, 'wb') as f:
        pickle.dump(final_bigrams, f)

    return features, labels


def extract_all_features():
    """
    Extracts all the features and saves them with their labels in pickles
    :return:
    """
    with open("data.pickle", 'rb') as f:
        data = pickle.load(f)
    pos_data = get_POS_data(data)
    with open("pos_bigrams_features.pickle", 'wb') as f:
        pickle.dump(pos_data, f)
    function_words_data = get_function_words_features(data)
    with open("function_words_features.pickle", 'wb') as f:
        pickle.dump(function_words_data, f)
    dependency_data = get_dependency_data(data)
    with open("dependency_bigrams_features.pickle", 'wb') as f:
        pickle.dump(dependency_data, f)


if __name__ == '__main__':
    extract_all_features()
