import os
import sys
import math
import numpy as np
import pandas as pd
import itertools as it
from pandas.core.frame import DataFrame
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
        self.size = len(alfabet) ** q
        self.grams = [["".join(p),0] for p in it.product(self.alfabet, repeat=q)]
        print('\n----------------------------------------------------\n')

        print(f"\n{q}-grams with {len(self.alfabet)} symbols in alfabet: {self.alfabet}")
        self.print_matrix(self.grams, len(self.alfabet)**q)

    def set_filein(self,path):
        self.filein = path

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

        freq = [ x / (len(seq) - q) for x in grams.values() ]
        gram = grams.keys()

        return gram,freq


def process_file(filename, file_type='fasta'):
    return [ [seq_record.id,str(seq_record.seq)] for seq_record in SeqIO.parse(filename, file_type) ]

def get_metadata(path='dataset/ncbi_dataset/data/sars_cov_2.csv'):
    print(f"Reading metadata from file: '{path}'\n")


    genomas = pd.read_csv(path)[['Nucleotide Accession','Geo Location']]
    genomas = genomas.rename(columns={'Nucleotide Accession':'label'})
    genomas['label'] = np.array((genomas['label']),'S20')
    genomas['continent'] = [ str(x).split(';')[0] for x in genomas['Geo Location'] ]
    genomas.loc[genomas['Geo Location'].isna(), 'continent'] = 'Not Reported'
    continents = list(set(genomas['continent']))
    genomas['target'] = genomas['continent'].apply(lambda x: continents.index(x))

    print(genomas.groupby(['target','continent'])[['label']].count().rename(columns={'label':'count'}))

    print('\n----------------------------------------------------\n')

    return genomas[['label','target','continent']]

def read_and_extract_frequences(qgram, path):

    print(f"Reading sequences from file: '{path}'\n")
    sequences = process_file(path)

    print(f'Extract {q}-grams frequence from all {len(sequences)} sequences read\n')

    data = np.zeros((len(sequences),qgram.size),float)
    label = np.zeros((len(sequences)),'S20')

    for i, (id, seq) in enumerate(tqdm(sequences)):
        seq = space(seq, qgram.alfabet)
        grams,freq = qgram.freqkgrams(seq,qgram.q)
        data[i] = freq
        label[i] = id

    dataset = pd.DataFrame(data,columns=grams)
    dataset['label'] = label

    genomas = get_metadata()
    dataset = pd.merge(dataset,genomas,how='left',on='label')


    target = dataset[['label','target','continent']]
    data = dataset.drop(columns=['label','target','continent']).to_numpy()

    print(target)
    print(data)

    filename = qgram.filein.split('/')[-1].split('.')[0] if '/' in qgram.filein else qgram.filein.split('\\')[-1].split('.')[0]

    if not os.path.isdir(f"extraction/{filename}"):
        os.makedirs(f"extraction/{filename}")

    np.savez_compressed(f"extraction/{filename}/{qgram.q}grams.npz",data=data,target=target, allow_pickle=True)

    print(f"Saved dataset data and target in 'extraction/{filename}/{qgram.q}grams.npz'")

    dataset[['label','target','continent']].to_csv(f"extraction/{filename}/{qgram.q}grams.csv")

    print(f"Saved meta informations from dataset in 'extraction/{filename}/{qgram.q}grams.csv'")

if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise Exception('USAGE: python ./qgram.py ACTG 2 ./dataset/ncbi_dataset/data/genomic.fna')

    alfabet = set(list(sys.argv[1]))
    q = int(sys.argv[2])
    qgram = KGram(alfabet,q)
    qgram.set_filein(sys.argv[3])
    read_and_extract_frequences(qgram,sys.argv[3])
