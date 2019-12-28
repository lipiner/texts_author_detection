The project contains the following files:
- preprocessing.py - extracts the data to list of texts and the corresponding list of labels (the names
    of the authors). The processed data is saved in data.pickle.
- features_extraction.py - given the data pickle, extracts 3 types of features to each text:
    function words frequencies, POS bigrams and dependency bigrams. The feature matrices
    are saved in features_pickles\ folder.
- main.py - runs the machine learning algorithms on the feature matrices, extracts graphs
    and results.