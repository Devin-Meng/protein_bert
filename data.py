# prof's data-pipline code for torch distributed learning
import h5py
import random
import numpy as np

from torch.utils.data import Dataset, Sampler


maxfraglen = 32*8
minseqlen, maxseqlen = 32*2, maxfraglen*2
ifn = '/home/Zhaoxu/Project/dataset/data_extract250_0000.hdf5'


class SeqSampler(Sampler):
    def __init__(self, idx, rank=0, world=1):
        self.idx = idx
        self.rank = rank
        self.world = world

        size = self.idx[1:] - self.idx[:-1]
        #size is a list contain length of sequences
        self.weight = np.zeros_like(size, dtype=np.float)
        self.weight[size >= minseqlen] = 1
        self.weight[size > maxfraglen] = size[size > maxfraglen] - maxfraglen
        self.weight[size > maxseqlen] = maxseqlen - maxfraglen
        self.weight = np.cumsum(self.weight)
        self.weight = self.weight / self.weight[-1]
        self.size = len(self.weight)
        #self.size is the number of sequences

    def __iter__(self):
        rndlst = list(range(self.size))
        #this is a list of sequence index
        while True:
            rndidx = random.choices(rndlst, cum_weights=self.weight, k=256*self.world)
            for i in range(self.rank, len(rndidx), self.world):
                i0, i1 = self.idx[rndidx[i]], self.idx[rndidx[i]+1]
                #i0, i1 is the start point and end point of the chosen sequence
                i1 = min(i1, i0 + maxseqlen)
                # truncate the sequence length to maxsequence length
                j0 = random.randint(i0, max(i0, i1 - maxfraglen))
                j1 = min(j0 + maxfraglen, i1)
                yield(i0, i1, j0, j1)

    def __len__(self):
        return self.size

class SeqDataset(Dataset):
    def __init__(self, seq, size):
        self.seq = seq
        self.size = size

    def __getitem__(self, i):
        begin, end = i[2] - i[0] + 1, i[1] - i[3] + 1  # position +1
        size = i[3] - i[2]
        # the length of revised version of sequence
        item1 = np.zeros(maxfraglen, dtype=np.int64)
        item1[:size] = self.seq[i[2]:i[3]] + 1  # class +1
        #item1 is the revised version sequence
        item0 = np.copy(item1)
        mask = np.zeros_like(item1, dtype=np.int8)
        for i, j in zip(random.choices(list(range(size)), k=size//7),
                random.choices(list(range(3)), cum_weights=[1/7, 2/7, 7/7], k=size//7)):
            if j == 0:    # unchange
                pass
            elif j == 1:  # change
                k = random.randint(1, 19)
                if k < item0[i]: item0[i] = k
                else: item0[i] = k + 1
            else:         # mask
                item0[i] = 22
            mask[i] = 1
        return item0, item1, mask, np.array([begin, end, size], dtype=np.int32)

    def __len__(self):
        return self.size

