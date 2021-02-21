
import sys
import numpy as np
import pandas as pd
from unidecode import unidecode as unidecode


def space(text):
    parts = filter(lambda x: len(x) > 0,text.split(" "))
    parts = unidecode("".join(parts))
    return list(filter(lambda x: x >= 'a' and x <= 'z', parts.lower()))

class KGram:
    def __len__(self):
        return len(self.__data)

    def __init__(self, data):
        self.__data = data
        self.__alefbet = sorted(set(data))

    def get_x(self):
        return self.__data
    
    def get_alefbet(self):
        return self.__alefbet
    
    def freqkgrams(self,k:int=1):
        ngrams = len(self.__alefbet)**k
        grams = dict()
        x = self.__data

        print(f"ngrams {ngrams}")

        for i in range(0,len(x)-k):
            window = "".join(x[i:i+k])
            if window in grams.keys():
                grams[window] += 1
            else:
                grams[window] = 1

        freq = np.fromiter(grams.values(),dtype=int)
        rfreq = freq / (len(x) - k)
        keys = list(grams.keys())
        grams = pd.DataFrame(freq,keys,columns=['freq'])
        grams['rfreq'] = rfreq
        return grams.sort_index()


filein = open(sys.argv[1],'r')
txt = space(filein.read())

sequence = KGram(txt)
print(f'sequence length {len(sequence)}')
print(f'{sys.argv[2]}-grams of {sequence.get_alefbet()}:')
freq = sequence.freqkgrams(int(sys.argv[2]))
print(freq.index.tolist())
print(freq)
print(len(freq))
print(freq.sum(axis=0))

