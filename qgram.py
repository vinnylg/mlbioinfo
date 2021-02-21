import os
import sys
import math
import numpy as np
import pandas as pd
import itertools as it
from unidecode import unidecode as unidecode
from Bio import SeqIO
from tqdm import tqdm

def space(text, alfabet):
    parts = filter(lambda x: len(x) > 0,text.split(" "))
    parts = unidecode("".join(parts))
    return list(filter(lambda x: x in alfabet, parts.upper()))

class KGram:
    def __len__(self):
        return len(self.__data)

    def __init__(self, alfabet: set, q: int):
        self.alfabet = sorted(alfabet)
        self.q = q
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

    def freqkgrams(self,seq: list,q:int=1):
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


def process_file(filename, file_type='fasta'):
    return [ [seq_record.id,str(seq_record.seq)] for seq_record in SeqIO.parse(filename, file_type) ]

def read_and_extract_frequences(qgram):
    if not os.path.isdir(f"extraction/{qgram.q}grams"):
        os.makedirs(f"extraction/{qgram.q}grams")


    print(f'Reading sequences from file: \'dataset/ncbi_dataset/data/genomic.fna\'')
    sequences = process_file('dataset/ncbi_dataset/data/genomic.fna')
    print(f'Extract {q}-grams frequence from all {len(sequences)} sequences read')

    for id, seq in tqdm(sequences):
        seq = space(seq, qgram.alfabet)
        freq = qgram.freqkgrams(seq,qgram.q)
        freq.index.name = 'grams'
        freq.to_csv(f"extraction/{qgram.q}grams/{id}.csv")

if __name__ == '__main__':

    alfabet = {'A','C','T','G'}
    q = int(sys.argv[1])
    qgram = KGram(alfabet,q)
    read_and_extract_frequences(qgram)
