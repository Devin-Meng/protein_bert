#sthis is preprocess file
#from the mounted path input masked sequence and form seq and idx list.
import argparse
import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser,add_argument("--mask_path")

a27 = 'ARNDCQEGHILKMFPSTWYVXUZJOB _'

ifn = args.mask_path
ofn = 'processed.hdf5'

seq_list, seq, idx = [], [], [0]
with open(ifn, 'r') as f:
    while True:
        line = f.readline()
        if line != '' and line[0] != '>':
            seq_list.append(line)
        if not line:
            break
    f.close()
print('file has been converted to list')

for i in seq_list:
    seq.append(np.array([a27.find(c) for c in i], dtype=np.int8))
    idx.append(idx[-1] + len(i))

seq = np.concatenate(seq)
seq[seq < 0] += len(a27)
idx = np.array(idx, dtype=np.int64)
print('processed')

with h5py.File(ofn, 'w') as f:
    f.create_dataset('seq', data=seq)
    f.create_dataset('idx', data=idx)
print('done!!!')
