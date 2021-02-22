import numpy as np

# dataset = np.load(,,allow_pickle=True)['dataset']

with np.load('extraction/2grams.npz','r', allow_pickle=True) as npfile:
    data = npfile['data'])
    target = npfile['target'])
    print(data)
    print(target)