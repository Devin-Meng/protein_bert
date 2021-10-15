#this is data_pipline file
#it should contain a sampler and a dataset class
#the sampler should truncate the sequence into fragment and take out in order
#the dataset should padding the fragment and form the final input
#remember idx[i +1] -1
import numpy as np
from torch.utils.data import Dataset, Sampler

fragment = 32*8

class SeqSampler(Sampler):
    def __init__(self, idx):
        self.idx = idx
        size = self.idx[1:] - self.idx[:-1]
        self.size = len(size)

    def __iter__(self):
        for i in range(self.size):
            i0, i1 = self.idx[i], self.idx[i + 1]
            j0 = i0
            j1 = min(i1, i0 + fragment)
            left = i1 -j1
            yield(i0, i1, j0, j1)
            while left > 0:
                j0 = j0 + fragment
                j1 = min(i1, j1 + fragment)
                left = i1 -j1
                yield(i0, i1, j0, j1)

    def __len__(self):
        return self.size

class SeqDataset(Dataset):
    def __init__(self, seq, size):
        self.seq = seq
        self.size = size

    def __getitem__(self, i):
        begin, end = i[2] - i[0] + 1, i[1] - i[3] + 1
        size = i[3] - i[2]
        item = np.zeros(fragment, dtype=np.int64)
        item[:size] = self.seq[i[2]:i[3]] + 1
        return item, i[1], np.array([begin, end, size], dtype=np.int32)

    def __len__(self):
        return self.size
