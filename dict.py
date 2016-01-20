import numpy as np

# Peter Norvig's most popular word list with word frequency http://norvig.com/ngrams/
dict1 = np.loadtxt("count_100k.txt", dtype=object)
dict2 = np.loadtxt("wordsEn.txt", dtype=object)

# Search dict1 for words in dict2
i = np.searchsorted(dict1[:, 0], dict2)
arr = dict1[i]

# Remove duplicate rows
_, u = np.unique(arr[:, 0], return_index=True)
arr = arr[u]

# Jungle Book names
names = np.loadtxt("names.txt", delimiter="/n", dtype=object)

# Add frequency for Jungle Book names
names_freq = np.empty((names.shape[0], 2), dtype=object)
names_freq[:, 0] = names
names_freq[:, 1] = int(np.max(arr, axis=0)[1]) + 1

# Create a frequency dictionary containing words and Jungle Book names
freq_list = np.vstack((arr, names_freq))
arr = freq_list[freq_list[:, 0].argsort()]

np.savetxt("combo.txt", arr, fmt="%s")
