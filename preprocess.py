import os

import h5py
import numpy as np
from Bio import SeqIO

ifn = '/home/Zhaoxu/Project/protein_bert/dataset'
ofn = '/home/Zhaoxu/Project/protein_bert/dataset/data_extract200_0000.hdf5'

extrat_number = 250_0000
seq_list, seq, idx = [], [], [0]
a27 = 'ARNDCQEGHILKMFPSTWYVXUZJOB '

for filename in os.scandir(ifn):
    if filename.path.endswith('.fasta') and eval(filename.path[-9: -6].lstrip('0')) <85:
        count = 0
        with open(filename) as handle:
            for seq_records in SeqIO.parse(handle, "fasta"):
                seq_list.append(str(seq_records.seq))
                count += 1
                if count > extrat_number:
                    print(count)
                    break
    print(f'{filename} has been processed')

# for seq_records in SeqIO.parse(test_ifn, 'fasta'):
#     seq_list.append(str(seq_records.seq))
# print('read sequences into list')

for i in seq_list:
    seq.append(np.array([a27.find(c) for c in i], dtype=np.int8))
    idx.append(idx[-1] + len(i))
print('processed')

del seq_list
seq = np.concatenate(seq)
seq[seq < 0] += len(a27)
idx = np.array(idx, dtype=np.int64)
print('concatenate numpy array')

print('#saving', ofn, '...')
with h5py.File(ofn, 'w') as f:
    f.create_dataset('seq', data=seq)
    f.create_dataset('idx', data=idx)
print("done!!!")
