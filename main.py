from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from os import path
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import pylab
import itertools

FEATURES_NUMS = [75, 150, 225, 300, 700, 1100, 1500, 1800]
RF_TREES_NUM = [50, 100, 200, 300]
RF_DEPTHS = range(6, 23, 4)
SVM_MODEL_NAME = "SVM"
RF_MODEL_NAME = "RF"
TITLE_SIZE = 32
LABEL_SIZE = 27
TICKS_SIZE = 19
PLOT_WIDTH = 18
FIGS_DIR = "figs"
COLORS = ['gray', 'rosybrown', 'red', 'coral', 'gold', 'olive', 'lawngreen', 'seagreen', 'lightblue',
          'deepskyblue', 'steelblue', 'navy', 'blue', 'blueviolet', 'plum', 'magenta', 'brown', 'lightpink',
          'yellowgreen', 'black', 'aquamarine', 'sandybrown', 'mediumslateblue', 'deeppink', 'c']


def perform_pca(X, features_num=None):
    """
    Performs PCA
    :param X: set of features
    :param features_num: number of features after permofrming PCA
    :return: the reduced set of features
    """
    if len(X) <= round(features_num) + 50:
        return X
    pca_model = PCA(n_components=features_num)
    features_reduced = pca_model.fit_transform(X)
    return features_reduced


def split_train_test_authors(features, labels):
    """
    Split the given features and labels sets to train set (0.6 from the data), validation set (0.2)
    and test test (0.2).
    This function ensures that there will be 60%-20%-20% of each label in the data in the train, validation and test
    sets respectively.
    :param features: features matrix
    :param labels: labels vector
    :return: train_features, val_features, test_features,train_labels, val_labels, test_labels
    """
    unique_labels = np.unique(labels)

    # initialize the vectors
    indices = labels == unique_labels[0]
    train_features, val_features, test_features, train_labels, val_labels, test_labels = split_train_validation_test(
        features[indices], labels[indices])

    for label in unique_labels[1:]:
        # get the indices of the current label
        indices = labels == label
        # split the data of the current label
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_validation_test(features[indices], labels[indices])
        # append the new split data
        train_features = np.append(train_features, X_train, axis=0)
        val_features = np.append(val_features, X_val, axis=0)
        test_features = np.append(test_features, X_test, axis=0)
        train_labels = np.append(train_labels, y_train, axis=0)
        val_labels = np.append(val_labels, y_val, axis=0)
        test_labels = np.append(test_labels, y_test, axis=0)

    return train_features, val_features, test_features, train_labels, val_labels, test_labels


def split_train_validation_test(features, labels):
    """
    split the data to train, validation and test sets (0.6, 0.2, 0.2)
    :param features: the features to split
    :param labels: the labels
    :return: X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4)
    # split the test data to validation and test data
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(clf, X_train, y_train, X_val, y_val, X_test, y_test, train_scores, val_scores, test_scores):
    """
    Trains the model and gets its scores.
    :param clf: a classifier
    :param X_train: train features set
    :param y_train: train labels
    :param X_val: validation features set
    :param y_val: validation labels
    :param X_test: test features set
    :param y_test: test labels
    :param train_scores: list of train scores for adding the current train score
    :param val_scores: list of validation scores for adding the current train score
    :param test_scores: list of test scores for adding the current train score
    """
    clf.fit(X_train, y_train)
    train_scores.append(clf.score(X_train, y_train))
    val_scores.append(clf.score(X_val, y_val))
    test_scores.append(clf.score(X_test, y_test))


def plot_RF_train_val(paradigm, models, train, val, features_num, keys, key_name, addition_key=None):
    """
    plot RF scores as function of the given key name
    :param paradigm: the current paradigm name that is being tested (for the title)
    :param models: all the models. list of tuples where the second index of the tuple is the model's name
    :param train: list of train scores (same order as the models)
    :param val: list of validation scores (same order as the models)
    :param features_num: current number of features (for the title)
    :param keys: list of keys for x axis
    :param key_name: the key name as it appears in the model's name (the legend will be with the model's name until
    the key name). Typically "depth" or "trees"
    :param addition_key: Optional. Additional key name to add at the end of the model name (the model name will
    appear with its name until the key name and than with the additional_key)
    """
    models_names = [model[1] for model in models]  # gets the models name
    # gets the data frame for the current plot
    train_df = special_df(models_names, train, "train", keys, key_name, RF_MODEL_NAME, addition_key)
    test_df = special_df(models_names, val, "val", keys, key_name, RF_MODEL_NAME, addition_key)

    title = "%s - RF train-validation accuracy Vs. %s for %d features" % (paradigm, key_name, features_num)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # setting the amount of colors in the plot to be the amount of models
    cm = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # ax.set_color_cycle(cm[:test_df.shape[1]])  # set the colors so it will have different color for each models
    ax.set_prop_cycle(color=cm[:test_df.shape[1]])

    test_df.plot(ax=ax)
    train_df.plot(ax=ax, style="--")
    if key_name == "trees":  # fix name for the x axis name
        key_name = "number"
    set_plt_params(ax, title, "Tree's %s" % key_name, "Accuracy")

    # saves the plot
    plt.savefig(path.join(FIGS_DIR, title + '.png'))
    plt.close()  # close the figure


def special_df(models_names, results, legend, keys, key_name, filtered_model="", addition_key=None):
    """
    Creates special dataframe where the index is the given keys, and the columns are the models name
    such that it contains the name only until the key name.
    :param models_names: all the models' names
    :param results: list of scores (same order as models_names)
    :param legend: legend's name. Will be added to the model's name (typically train/test...)
    :param keys: list of keys for x axis
    :param key_name: the key name as it appears in the model's name (the legend will be with the model's name until
    the key name). Typically "depth" or "trees"
    :param filtered_model: a string to filter on so only models containing this string will be in the dataframe
    :param addition_key: Optional. Additional key name to add at the end of the model name (the model name will
    appear with its name until the key name and than with the additional_key)
    :return: the final dataframe contains
    """
    results = {name: results[i] for i, name in enumerate(models_names) if filtered_model in name}
    final_results = dict()
    # creates dictionary for the data frame. Each key contains its scores
    for key in keys:
        for name in results:
            if str(key) in name:
                # create the results only for the current key
                # fix the model name so it won't contain the key (so it will be same names regardless only this key)
                name_parts = name.split("_")
                end_name_idx = name_parts.index(key_name) - 1
                df_name = "_".join(name_parts[:end_name_idx])
                if addition_key:
                    df_name += "_" + "_".join(
                        name_parts[name_parts.index(addition_key) - 1:name_parts.index(addition_key) + 1])
                df_name += " " + legend
                # add the result
                if final_results.get(key) is None:
                    final_results[key] = dict()
                final_results[key][df_name] = results[name]
    return pd.DataFrame(final_results).transpose().sort_index()


def set_plt_params(ax, title, x_label, y_label, legend=True, width=PLOT_WIDTH):
    """
    Set the plot parameters (size, labels and etc.)
    :param ax: the plot axis
    :param title: the plot's title
    :param x_label: the plot's x label
    :param y_label: the plot's y label
    :param legend: whether or not to show the legend. Default: True
    :param width: the plot's width
    :return:
    """
    ax.set_ylabel(y_label, fontsize=LABEL_SIZE)
    ax.set_xlabel(x_label, fontsize=LABEL_SIZE)
    if legend:
        plt.legend(fontsize=TICKS_SIZE)
    plt.grid(axis='y')

    plt.title(title, fontsize=TITLE_SIZE, y=1.03)

    plt.xticks(fontsize=TICKS_SIZE)
    plt.yticks(fontsize=TICKS_SIZE)
    fig = plt.gcf()
    fig.set_size_inches((width, 10), forward=False)


def plot_features_train_val(paradigm, models, train, val):
    """
    plot RF scores as function of the given key name
    :param paradigm: the current paradigm name that is being tested (for the title)
    :param models: all the models. list of tuples where the second index of the tuple is the model's name
    :param train: list of train scores (same order as the models)
    :param val: list of validation scores (same order as the models)
    """
    title = "%s_train_validation_results" % paradigm
    models_names = [model[1] for model in models]  # gets the models name
    # gets the data frame for the current plot
    train = special_df(models_names, train, "train", FEATURES_NUMS, "features")
    if len(train) <= 1:
        # no data to plot (no 2 different values for x axis)
        return
    test = special_df(models_names, val, "val", FEATURES_NUMS, "features")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # set the plot's colors so it will have different color for each model
    ax.set_prop_cycle(color=COLORS[:test.shape[1]])

    test.plot(ax=ax)
    train.plot(ax=ax, style="--")
    set_plt_params(ax, title.replace("_", " "), "Number of features", "Accuracy")

    # saves figure legend as a different file
    fig_legend = pylab.figure(figsize=(4, 10))
    pylab.figlegend(*ax.get_legend_handles_labels(), loc='center')
    fig_legend.savefig(path.join(FIGS_DIR, title + '_legend.png'))
    ax.get_legend().remove()

    fig.savefig(path.join(FIGS_DIR, title + '.png'))
    plt.close()  # close the figure


def calc_std(models, test_sets):
    """
    Calculates the std of the test scores of the accuracies on each author, and plots accuracy as number of texts in
    the dataset
    :param models: the model of each paradigm
    :param test_sets: the test set of each model
    """
    stds = dict()
    filtered_stds = dict()
    for i, filename in enumerate([
        "pos_bigrams_features.pickle",
        "dependency_bigrams_features.pickle",
        "function_words_features.pickle",
    ]):
        paradigm = filename[:-len("_features.pickle")].replace("_", " ")
        print("pardigm: %s" % paradigm)

        # load the features and labels
        with open(path.join("features_pickles", filename), 'rb') as f:
            _, labels = pickle.load(f)

        clf = models[i]
        test_features, test_labels = test_sets[i]
        text_num_scores = dict()  # accuracy score for each labels' occurrences
        # calculate the accuracy for each label
        for label in np.unique(test_labels):
            texts_num = np.count_nonzero(labels == label)

            indices = np.where(test_labels == label)[0]
            score = clf.score(test_features[indices], test_labels[indices])  # accuracy score for current label
            if texts_num not in text_num_scores:  # this number of texts not appears yet in the dictionary
                text_num_scores[texts_num] = list()
            text_num_scores[texts_num].append(score)  # adds the current score for the number of texts

        # calculates the average score for each label's occurrences number
        averaged_scores = {num: np.mean(text_num_scores[num]) for num in text_num_scores}
        # plot the accuracy as function of number of texts
        averaged_scores_df = pd.DataFrame.from_dict(averaged_scores, orient='index')
        averaged_scores_df = averaged_scores_df.sort_index()
        averaged_scores_df.plot(title="%s - accuracy as function of number of texts" % paradigm, legend=False)
        plt.ylabel("Accuracy")
        plt.xlabel("Number of texts")
        plt.grid(axis='y')
        plt.savefig(path.join(FIGS_DIR, 'authors_scores_%s.png' % paradigm))
        plt.close()  # close the figure

        authors_scores = list(itertools.chain(*list(text_num_scores.values())))  # gets all the scores per label
        stds[paradigm] = np.std(authors_scores)  # the paradigm's std over all authors

        authors_filtered_scores = list(
            itertools.chain(*list({num: text_num_scores[num] for num in text_num_scores if num >= 20}.values())))
        filtered_stds[paradigm] = np.std(authors_filtered_scores)

    print("stds:", stds)
    print("filtered stds:", filtered_stds)


def main():
    best_scores = dict()
    best_models = list()
    best_test_sets = list()
    for filename in [
        "pos_bigrams_features.pickle",
        "dependency_bigrams_features.pickle"
        "function_words_features.pickle",
    ]:
        paradigm = filename[:-len("_features.pickle")].replace("_", " ")
        print("pardigm: %s" % paradigm)

        # load the features and labels
        with open(path.join("features_pickles", filename), 'rb') as f:
            features, labels = pickle.load(f)

        models = list()
        train_scores = list()
        val_scores = list()
        test_scores = list()
        all_test_sets = list()
        for features_num in FEATURES_NUMS:
            curr_models = list()
            curr_train_scores = list()
            curr_val_scores = list()
            if features_num > features.shape[1]:
                break  # equals to previous iteration. All features were already used
            print("CURR: %d" % features_num)

            # performs PCA and split the data to train, validation and test sets
            curr_features = perform_pca(features, features_num)
            train_features, val_features, test_features, train_labels, val_labels, test_labels = \
                split_train_test_authors(curr_features, labels)
            all_test_sets.append((test_features, test_labels))

            # prepare the models
            curr_models += [(svm.LinearSVC(max_iter=2000), SVM_MODEL_NAME + "_%d_features" % features_num)]  # SVM
            curr_models += [(RandomForestClassifier(n_estimators=i, max_depth=j),
                            RF_MODEL_NAME + "_%d_trees_%d_depth_%d_features" % (i, j, features_num))
                            for i in RF_TREES_NUM for j in RF_DEPTHS]  # Random Forest

            # performs learning algorithms
            for clf, clf_name in curr_models:
                print("model: %s" % clf_name)
                train_model(clf, train_features, train_labels, val_features, val_labels, test_features, test_labels,
                            curr_train_scores, curr_val_scores, test_scores)
            plot_RF_train_val(paradigm, curr_models, curr_train_scores, curr_val_scores, features_num, RF_DEPTHS, "depth")
            plot_RF_train_val(paradigm, curr_models, curr_train_scores, curr_val_scores, features_num, RF_TREES_NUM,
                              "trees", "depth")
            models.extend(curr_models)
            train_scores.extend(curr_train_scores)
            val_scores.extend(curr_val_scores)

        # saves the data
        with open("models_pickles/%s_scores.pickle" % paradigm, 'wb') as f:
            pickle.dump((train_scores, val_scores, test_scores), f)

        # get test score of the best model from the validation scores
        best_clf_idx = np.argmax(val_scores)
        best_model, best_clf_name = models[best_clf_idx]
        test_score = test_scores[best_clf_idx]
        best_scores[paradigm] = (best_clf_name, test_score)
        # saves the best model and set for later use
        best_models.append(best_model)
        models_num = len(models) // len(FEATURES_NUMS)  # calculates the number of different models in each feature num
        best_test_sets.append(all_test_sets[best_clf_idx // models_num])  # adds the relevant test set

        # plots the features trade off for the current paradigm
        plot_features_train_val(paradigm, models, train_scores, val_scores)

    # plots and prints the test scores
    best_scores_df = pd.DataFrame.from_dict({paradigm: best_scores[paradigm][1] for paradigm in best_scores.keys()},
                                            orient='index')
    best_scores_df.plot(kind='bar', title="Test scores", legend=False, rot=0)
    plt.ylabel("Accuracy")
    plt.xlabel("Features types")
    plt.grid(axis='y')
    plt.savefig(path.join(FIGS_DIR, 'test scores.png'))
    plt.close()  # close the figure

    print(best_scores)
    calc_std(best_models, best_test_sets)


if __name__ == '__main__':
    main()
