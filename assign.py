import numpy as np
import scipy
from scipy import linalg
from bisect import bisect_left


def load_data_array(filename):
    """
    Load the data file containing the labels and bounding box info
    :param filename: The data file to be loaded
    :return: data, labels - data: [n, 5] matrix containing the bounding box and word ends data for the n labels,
            and a list of the labels converted to an int value
    """
    array = np.loadtxt(filename, dtype=object)

    # Converts the char labels into an integer value
    labels = [ord(x) - 65 for x in array[:, 0]]
    data = array[:, 1:].astype(np.float32)

    return data, np.asarray(labels)


def get_character(page, pos):
    """
    Gets the character from the page specified by the bounding box
    :param page: The page to load the character from
    :param pos: The bounding box for the character
    :return: matrix representing the character
    """
    # Image is flipped in y-axis
    top = page.shape[0] - pos[3]
    bottom = page.shape[0] - pos[1]

    return page[top:bottom, pos[0]:pos[2]]


def covert_to_labels(words):
    """
    Converts words into the integer labels for score computation
    :param words: List of words to be converted
    :return: List of integer labels
    """
    return np.asarray([ord(char) - 65 for word in words for char in word])


def covert_to_words(labels, word_ends):
    """
    Converts a list of integer labels into the list of words they represent
    :param labels: Labels to be converted
    :param word_ends: List of word ends from the page data file
    :return: List of words
    """
    cur_word = ""
    words = []

    for i in xrange(labels.shape[0]):
        char = chr(labels[i] + 65)
        cur_word += char

        if word_ends[i] == 1:
            words.append(cur_word)
            cur_word = ""

    return words


def get_max_feature_size(dat):
    """
    Gets the max size of the characters on a page
    :param dat: Page data file
    :return: max size - [w, h]
    """
    array = np.zeros((dat.shape[0], 2))

    # Calculate the size of each character
    array[:, 0] = dat[:, 3] - dat[:, 1]
    array[:, 1] = dat[:, 2] - dat[:, 0]

    # Return the maximum of both columns (width and height)
    return array.max(axis=0)


def get_feature_matrix(page, data, feature_size):
    """
    Turns every character on the page into a feature vector of a certain size and returns all as a matrix
    :param page: The page to be converted into feature vectors
    :param data: The bounding box data for the page
    :param feature_size: The required size of the feature vectors, must be >= the size of the characters on the page
    :return: A matrix of feature vectors for every character on the page
    """

    feature_matrix = np.zeros((data.shape[0], np.prod(feature_size)))

    for i in xrange(data.shape[0]):
        chara = get_character(page, data[i])

        # Padding - the image is padded right and bottom to get to the required size
        zeros = np.zeros(feature_size)
        zeros[:chara.shape[0], :chara.shape[1]] = chara

        # Reshape as a vector and add to the matrix
        feature_matrix[i] = zeros.reshape(np.prod(feature_size))

    return feature_matrix


def classify(train, train_labels, test, features=None):
    """Nearest neighbour classification.

    :param train data matrix storing training data, one sample per row
    :param train_labels: a vector storing the training data labels
    :param test:  data matrix storing the test data
    :param features a vector of indices that select the feature to use
             if features=None then all features are used

    :return:  (score, confusions) - a percentage correct and a
                                  confusion matrix
    """

    # Use all feature is no feature parameter has been supplied
    if features is None:
        features = np.arange(0, train.shape[1])

    # Select the desired features from the training and test data
    train = train[:, features]
    test = test[:, features]

    # Super compact implementation of nearest neighbour
    x = np.dot(test, train.transpose())
    modtest = np.sqrt(np.sum(test*test, axis=1))
    modtrain = np.sqrt(np.sum(train*train, axis=1))
    dist = x/np.outer(modtest, modtrain.transpose())  # cosine distance
    nearest = np.argmax(dist, axis=1)
    # mdist = np.max(dist, axis=1)
    labels = train_labels[0, nearest]

    return labels


def classification_pca(train_data, train_labels, test_data, test_labels, word_ends, num_dimens):
    """
    Takes the classifier inputs, creates the pca features, classifies using
    nearest neighbour, error corrects then calculates the percentage correct
    :param train_data: training data for the classifier
    :param train_labels: training labels for the classifer
    :param test_data: data to be classified
    :param test_labels: test labels for evaluation
    :param word_ends: vector of word ends for test data to use in error correction
    :param num_dimens: number of pca dimensions to be used
    :return: list of the classified and corrected words
    """
    # Create pca features
    cov = np.cov(train_data, rowvar=0)
    n = cov.shape[0]
    w, v = scipy.linalg.eigh(cov, eigvals=(n-num_dimens, n-1))
    v = np.fliplr(v)

    pcatrain_data = np.dot((train_data - np.mean(train_data)), v)
    pcatest_data = np.dot((test_data - np.mean(train_data)), v)

    # Classify using the nearest neighbour classifier
    classified = classify(pcatrain_data, train_labels.reshape((1, train_labels.shape[0])),
                          pcatest_data, xrange(0, num_dimens))

    # Error correction and percentage calculation
    guessed_words = covert_to_words(classified, word_ends)
    corrected_words = error_correction(guessed_words)
    corrected_labels = covert_to_labels(corrected_words)
    post_score = (100.0 * sum(test_labels == corrected_labels))/corrected_labels.shape[0]

    print("Score:", post_score, "%")
    print("\n")
    print(" ".join(corrected_words))
    print("\n")

    return corrected_words


def index_sorted(sorted_list, x):
    """
    Binary search on a sorted list
    :param sorted_list: sorted list to be searched
    :param x: element to be checked for
    :return: position of x in list or -1 if x not in list
    """
    i = bisect_left(sorted_list, x)
    if len(sorted_list) != i and sorted_list[i] == x:
        return i
    else:
        return -1


def error_correction(words):
    """
    Corrects words using an edit distance and word frequency based technique
    :param words: List of words that have been classified
    :return: List of words including those that have been corrected
    """
    freq_list = np.loadtxt("combo.txt", dtype=object)
    english_words = freq_list[:, 0]

    # Get a list of incorrect words by checking against a dictionary
    # using binary search - set() creates a unique list
    incorrect_words = [word.lower() for word in set(words)
                       if index_sorted(english_words, word.lower()) == -1]

    for incorrect in incorrect_words:
        # Vector of the possible correct words
        candidates = edit(incorrect, english_words)

        if candidates.shape[0] > 0:
            arr = np.empty((candidates.shape[0], 2), dtype=object)

            # Get the indices of the words in the english list so we can get the frequency
            i = np.searchsorted(english_words, candidates[:, 0])
            arr[:, 0] = english_words[i]

            # Calculate the probability of the candidate words: p = word_freq / (distance^15)
            arr[:, 1] = freq_list[i, 1].astype(float)/pow(candidates[:, 1].astype(float), 15)

            # Choose the candidate word with the highest p and replace in the word list
            replace_word = arr[np.argmax(arr, axis=0)[1], 0]

            words = [replace_word.title() if incorrect.title() == orig else replace_word
                     if incorrect == orig.lower() else orig for orig in words]

    return words


def edit(word, dictionary):
    """
    Creates a list of candidate words with 1 or 2 substitutions from the original
    :param word: word to create candidates for
    :param dictionary: dictionary to check candidate words against
    :return: vector of candidate words [word, distance] that are in the dictionary
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    words = []

    for i in xrange(len(word)):
        # Single substitution
        words += [(word[:i] + char + word[i+1:], 1) for char in alphabet if char != word[i]]

        for j in xrange(i+1, len(word)):
            # Two substitutions
            words += [(word[:i] + char1 + word[i+1:j] + char2 + word[j+1:], 2)
                      for char1 in alphabet for char2 in alphabet
                      if char1 != word[i] if char2 != word[j]]

    # Filters out candidates that aren't in the dictionary
    candidates = [[x, edits] for x, edits in set(words) if index_sorted(dictionary, x) != -1]
    return np.asarray(candidates)


def run():
    train1_dat, train1_labels = load_data_array("train1.dat")
    train2_dat, train2_labels = load_data_array("train2.dat")
    train3_dat, train3_labels = load_data_array("train3.dat")
    train4_dat, train4_labels = load_data_array("train4.dat")
    train_labels = np.hstack((train1_labels, train2_labels, train3_labels, train4_labels))

    test1_dat, test1_labels = load_data_array("test1.dat")
    test2_dat, test2_labels = load_data_array("test2.dat")

    # Get the max feature size from bounding box data
    max_size = get_max_feature_size(np.vstack((train1_dat, train2_dat, train3_dat,
                                               train4_dat, test1_dat, test2_dat)))

    train1_page = np.load("train1.npy")
    train2_page = np.load("train2.npy")
    train3_page = np.load("train3.npy")
    train4_page = np.load("train4.npy")

    # Create the training feature vectors using max feature size
    train1_data = get_feature_matrix(train1_page, train1_dat, max_size)
    train2_data = get_feature_matrix(train2_page, train2_dat, max_size)
    train3_data = get_feature_matrix(train3_page, train3_dat, max_size)
    train4_data = get_feature_matrix(train4_page, train4_dat, max_size)
    train_data = np.vstack((train1_data, train2_data, train3_data, train4_data))

    # Loop through pca dimensions
    for dimens in [40, 10]:
        # Loop through test pages
        for test in xrange(1, 3):
            if test == 1:
                test_labels = test1_labels
                test_dat = test1_dat
            else:
                test_labels = test2_labels
                test_dat = test2_dat

            # Loop through test image quality
            for page in xrange(0, 5):
                if page == 0:
                    test_page = np.load("test" + str(test) + ".npy")
                else:
                    test_page = np.load("test" + str(test) + "." + str(page) + ".npy")
                test_data = get_feature_matrix(test_page, test_dat, max_size)

                # Classify each page
                print("test" + str(test) + "." + str(page) + ", " + str(dimens)
                      + " dimensions")
                classification_pca(train_data, train_labels, test_data, test_labels,
                                   test_dat[:, 4], dimens)

run()
