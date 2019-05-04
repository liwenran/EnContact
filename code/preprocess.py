import sys

filepath = sys.argv[1]
RESIZED_LEN = 2000

from Bio import SeqIO
import os
import pandas as pd
import numpy as np

fasta_sequence_dict = SeqIO.to_dict(SeqIO.parse(open('hg19.fa'), 'fasta'))

locations = pd.read_csv(filepath, sep = '\t', names = ['chr', 'start', 'stop', 'label'])
promoter_A = pd.DataFrame(index = range(0, locations.shape[0]),
                          columns = ['chr', 'original_start', 'original_end',
                                     'resized_start', 'resized_end', 'resized_sequence',
                                     'label', 'index'])
for i in range(0, locations.shape[0]):
    chromosome = locations['chr'][i]
    promoter_A['chr'][i] = chromosome
    original_location = (locations['start'][i], locations['stop'][i])
    promoter_A['original_start'][i] = original_location[0]
    promoter_A['original_end'][i] = original_location[1]
    original_len = original_location[1] - original_location[0]
    len_difference = RESIZED_LEN - original_len
    resized_start = original_location[0] - len_difference / 2
    resized_end = resized_start + RESIZED_LEN
    promoter_A['resized_start'][i] = resized_start
    promoter_A['resized_end'][i] = resized_end
    promoter_A['resized_sequence'][i] = str(fasta_sequence_dict[chromosome].seq[resized_start : resized_end])
    promoter_A['label'][i] = locations['label'][i]
    promoter_A['index'][i] = i
promoter_A.to_csv(filepath[:-4]+'_A.csv', index = False)

FOLD = 10
enhancer_A = pd.read_csv(cell+'/enhancer1_A.csv')
promoter_A = pd.read_csv(cell+'/enhancer2_A.csv')
n_sample = enhancer_A.shape[0]
rand_index = range(0, n_sample)
np.random.seed(n_sample)
np.random.shuffle(rand_index)
n_sample_B = n_sample - n_sample // FOLD
enhancer_B = enhancer_A.iloc[rand_index[:n_sample_B]]
enhancer_C = enhancer_A.iloc[rand_index[n_sample_B:]]
promoter_B = promoter_A.iloc[rand_index[:n_sample_B]]
promoter_C = promoter_A.iloc[rand_index[n_sample_B:]]
with open('rand_index.pkl', 'wb') as f:
    pickle.dump(rand_index, f)
enhancer_B.to_csv(cell+'/enhancer1_B.csv', index = False)
enhancer_C.to_csv(cell+'/enhancer1_C.csv', index = False)
promoter_B.to_csv(cell+'/enhancer2_B.csv', index = False)
promoter_C.to_csv(cell+'/enhancer2_C.csv', index = False)

seq_dict = {'A':[1, 0, 0, 0], 'G':[0, 1, 0, 0],
            'C':[0, 0, 1, 0], 'T':[0, 0, 0, 1],
            'a':[1, 0, 0, 0], 'g':[0, 1, 0, 0],
            'c':[0, 0, 1, 0], 't':[0, 0, 0, 1]}

def seq_to_one_hot(filename):
    data = pd.read_csv(filename)
    label = np.array(data['label'])
    n_sample = data.shape[0]
    one_hot_list = []
    for i in range(0, n_sample):
        temp = []
        for c in data['resized_sequence'].iloc[i]:
            temp.extend(seq_dict.get(c, [0, 0, 0, 0]))
        one_hot_list.append(temp)
    sequence = np.array(one_hot_list, dtype='float32')
    filename = filename.split('.')[0] + '.npz'
    np.savez(filename, label = label, sequence = sequence)

enhancer_files = ['enhancer1_B.csv', 'enhancer1_C.csv']
promoter_files = ['enhancer2_B.csv', 'enhancer2_C.csv']
for filename in enhancer_files + promoter_files:
    seq_to_one_hot(cell+'/'+filename)
    print filename

