import sys,os
from Bio import SeqIO
import pandas as pd
import numpy as np

def generate_sequences():
	filepath = sys.argv[1]
	RESIZED_LEN = 2000
	fasta_sequence_dict = SeqIO.to_dict(SeqIO.parse(open('hg19.fa'), 'fasta'))

	locations = pd.read_csv(filepath, sep = '\t', names = ['chr', 'start', 'stop', 'label'])
	enhancer = pd.DataFrame(index = range(0, locations.shape[0]),
							  columns = ['chr', 'original_start', 'original_end',
										 'resized_start', 'resized_end', 'resized_sequence',
										 'label', 'index'])
	for i in range(0, locations.shape[0]):
		chromosome = locations['chr'][i]
		enhancer['chr'][i] = chromosome
		original_location = (locations['start'][i], locations['stop'][i])
		enhancer['original_start'][i] = original_location[0]
		enhancer['original_end'][i] = original_location[1]
		original_len = original_location[1] - original_location[0]
		len_difference = RESIZED_LEN - original_len
		resized_start = original_location[0] - len_difference / 2
		resized_end = resized_start + RESIZED_LEN
		enhancer['resized_start'][i] = resized_start
		enhancer['resized_end'][i] = resized_end
		enhancer['resized_sequence'][i] = str(fasta_sequence_dict[chromosome].seq[resized_start : resized_end])
		enhancer['label'][i] = locations['label'][i]
		enhancer['index'][i] = i
	enhancer.to_csv(filepath[:-4]+'_A.csv', index = False)
	return
	
def split_data():	
	FOLD = 10
	enhancer1 = pd.read_csv(cell+'/enhancer1_A.csv')
	enhancer2 = pd.read_csv(cell+'/enhancer2_A.csv')
	n_sample = enhancer1.shape[0]
	rand_index = range(0, n_sample)
	np.random.seed(n_sample)
	np.random.shuffle(rand_index)
	n_sample_B = n_sample - n_sample // FOLD
	enhancer1_B = enhancer1.iloc[rand_index[:n_sample_B]]
	enhancer1_C = enhancer1.iloc[rand_index[n_sample_B:]]
	enhancer2_B = enhancer2.iloc[rand_index[:n_sample_B]]
	enhancer2_C = enhancer2.iloc[rand_index[n_sample_B:]]
	with open('rand_index.pkl', 'wb') as f:
		pickle.dump(rand_index, f)
	enhancer1_B.to_csv(cell+'/enhancer1_B.csv', index = False)
	enhancer1_C.to_csv(cell+'/enhancer1_C.csv', index = False)
	enhancer2_B.to_csv(cell+'/enhancer2_B.csv', index = False)
	enhancer2_C.to_csv(cell+'/enhancer2_C.csv', index = False)
	return

def seq_to_one_hot(filename):
	seq_dict = {'A':[1, 0, 0, 0], 'G':[0, 1, 0, 0],
				'C':[0, 0, 1, 0], 'T':[0, 0, 0, 1],
				'a':[1, 0, 0, 0], 'g':[0, 1, 0, 0],
				'c':[0, 0, 1, 0], 't':[0, 0, 0, 1]}
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
