import os
import sys
import math
import numpy as np
from numpy.lib.function_base import append
import pandas as pd
import itertools as it
from unidecode import unidecode as unidecode


def space(text, alfabet):
    parts = filter(lambda x: len(x) > 0,text.split(" "))
    parts = unidecode("".join(parts))
    return list(filter(lambda x: x in alfabet, parts.upper()))

class KGram:
    def __len__(self):
        return len(self.__data)

    def __init__(self, alfabet: set, q: int):
        self.alfabet = sorted(alfabet)
        self.grams = [("".join(p),0) for p in it.product(self.alfabet, repeat=q)]
        print('\n----------------------------------------------------\n')

        print(f"\n{q}-grams with {len(self.alfabet)} symbols in alfabet: {self.alfabet}")
        self.print_matrix(self.grams, len(self.alfabet)**q)


    def print_matrix(self, matrix, size):
        rows = int(math.sqrt(size))
        columns = rows

        print(f"M|n**q| = {len(matrix)}\n")

        for i in range(rows):
            for j in range(columns):
                print(matrix[i*rows+j],end='\t')
            print('')

        print('\n----------------------------------------------------\n')

    def print_seq(self, x: list, n: int):
        print(f"\nsequence:\t{''.join(x[0:n])}...{''.join(x[len(x)-n:len(x)])}\n")
        print(f"x {len(x)} elements in alfabet {self.alfabet}\n")

    def freqkgrams(self,seq: list,q:int=1):
        self.print_seq(seq,10)

        grams = dict(self.grams)
        for i in range(0,len(seq)-q):
            window = "".join(seq[i:i+q])
            if window in grams.keys():
                grams[window] += 1
            else:
                grams[window] = 1

        grams = pd.DataFrame(grams.values(),index=grams.keys(), columns=['count'])
        grams['freq'] = grams['count'] / (len(seq) - q)
        return grams.sort_index()

def read_and_extract_frequences(qgream):
    for root, _, files in os.walk('dataset'):
        for filename in files:
            filein = f"{root}/{filename}"
            print(f"file: {filein}")

            with open(filein,'r') as seq:
                seq = space(seq.read(), alfabet)
                freq = qgram.freqkgrams(seq,q)
                freq.index.name = 'grams'
                freq.to_csv(f"extraction/{filename.split('.')[0]}.csv")

            print(freq)
            print('\n----------------------------------------------------\n')

if __name__ == '__main__':

    alfabet = {'A','C','T','G'}
    q = int(sys.argv[1])
    qgram = KGram(alfabet,q)
    read_and_extract_frequences(qgram)



