import random,sys
import numpy as np
from itertools import combinations, product

def get_Einfo():
    Einfo = {}
    Chrom = {}
    fin = open('human_permissive_enhancers.bed')
    for line in fin:
        line = line.strip().split('\t')
        Einfo[line[3]] = line[1]
        if line[0] not in Chrom:
            Chrom[line[0]] = line[3]
        else:
            Chrom[line[0]] += '\t'+line[3]
    fin.close()
    return Einfo, Chrom
Einfo, Chrom_enhancer = get_Einfo()

def get_candidates():
    """"get all possible distances within chroms except pos_distances"""
    AllCombs = []
    for ch in ['chr'+str(i) for i in range(1,23)]:
        enhancers = Chrom_enhancer[ch].split('\t')
        for e in list(combinations(enhancers, 2)):
            AllCombs.append(e)

    Candidates = list(set(AllCombs).difference(set(pos_distances.keys())))
    candidate_distances = {}
    for genes in Candidates:
        candidate_distances[genes] = np.abs(int(Einfo[genes[0]]) - int(Einfo[genes[1]]))
    print('The num of candidate pairs within chromsome is {}.'.format(len(candidate_distances)))
    return candidate_distances

def get_neg_distances():
	pos_values = sorted(pos_distances.values())
	num = len(pos_values)
	each_bin_num = round(num/5)
	neg_num = RATIO*int(num)
	print num, each_bin_num, neg_num
	#bins
	bin0 = pos_values[0]
	bin1 = pos_values[int(1*each_bin_num)]
	bin2 = pos_values[int(2*each_bin_num)]
	bin3 = pos_values[int(3*each_bin_num)]
	bin4 = pos_values[int(4*each_bin_num)]
	bin5 = pos_values[num-1]
	print bin0,bin1,bin2,bin3,bin4,bin5
	candidate_distances = sorted(candidate_distances.items, key=lambda d:d[1]) #ascending
	Bins = [ [] for row in range(5) ]
	for genes in candidate_distances:
		if bin0<candidate_distances[genes]  <=bin1:
			Bins[0].append(genes)
		elif bin1<candidate_distances[genes]<=bin2:
			Bins[1].append(genes)
		elif bin2<candidate_distances[genes]<=bin3:
			Bins[2].append(genes)
		elif bin3<candidate_distances[genes]<=bin4:
			Bins[3].append(genes)
		elif bin4<candidate_distances[genes]<=bin5:
			Bins[4].append(genes)
	#sampling
	neg_pairs = []
	for i in range(5):
		print('The num of candidate pairs in Bin {} is {}.'.format(i, len(Bins[i])))
		index = range(len(Bins[i]))
		random.shuffle(index)
		[neg_pairs.append(Bins[i][j]) for j in index[:int(neg_num/5)]]
	print('The num of negative pairs within chromsomes is {}.'.format(len(neg_pairs)))
	#save
	fout = open(cell+'/'+name+'.neg.pair-Rdist','w')
	for pair in neg_pairs:
		fout.write(pair[0]+'\t'+pair[1]+'\n')
	fout.close()
	print('The total num of negative pairs is {}.'.format(len(neg_pairs)))

def get_neg_rand():
    file = open(cell+'/'+cell+'_1v1_pair.txt')
    pos_lines = file.readlines()
    NUM_POS = len(pos_lines)

    #random pairs (RPair)
    fout = open(cell+'/'+name+'.neg.pair-Rpair','w')
    for i in range(NUM_POS*RATIO):
        E1 = random.sample(Einfo.keys(), 1)[0]
        E2 = random.sample(Einfo.keys(), 1)[0]
        fout.write(E1+'\t'+E2+'\n')
    fout.close()

    #random targets(REnhancer)
    fout = open(cell+'/'+name+'.neg.pair-Rside','w')
    for line in pos_lines:
        E1 = line.strip().split('\t')[0]
        for i in range(RATIO):
            E2 = random.sample(Einfo.keys(), 1)[0]
            fout.write(E1+'\t'+E2+'\n')
    fout.close()

def GetEnhancerInfo():
    infofile=open('human_permissive_enhancers.bed')
    EnhancerInfo={}
    for line in infofile:
        line=line.strip().split('\t')
        EnhancerInfo[line[3]]=(line[0], line[1], line[2])
    return EnhancerInfo

def readPairs(filename,fout,label):
    fin = open(filename)
    for line in fin:
        line=line.strip().split('\t')
        (c1,s1,e1) = EnhancerInfo[line[0]]
        (c2,s2,e2) = EnhancerInfo[line[1]]
        fout.write('\t'.join([c1,s1,e1, line[0], c2,s2,e2, line[1], str(label)])+'\n')


"""main"""
cell = sys.argv[1]

print('get_candidates...')
candidate_distances = get_candidates()
print('get_neg_distances...')
get_neg_distances()
get_neg_rand()
print('Done!')

EnhancerInfo = GetEnhancerInfo()
fout = open(cell+'/'+name+'.PN-Rdist.bed', 'w')
readPairs(cell+'/'+cell+'_1v1_pair.txt', fout, 1)
readPairs(cell+'/'+name+'.neg.pair-Rdist', fout, 0)
fout.close()

EnhancerInfo = GetEnhancerInfo()
fout = open(cell+'/'+name+'.PN-Rside.bed', 'w')
readPairs(cell+'/'+cell+'_1v1_pair.txt', fout, 1)
readPairs(cell+'/'+name+'.neg.pair-Rside', fout, 0)
fout.close()

EnhancerInfo = GetEnhancerInfo()
fout = open(cell+'/'+name+'.PN-Rpair.bed', 'w')
readPairs(cell+'/'+cell+'_1v1_pair.txt', fout, 1)
readPairs(cell+'/'+name+'.neg.pair-Rpair', fout, 0)
fout.close()

