import os
import argparse
from collections import Counter
from data_pipline import SeqSampler, SeqDataset
from model import Transformer
from torch.utils.data import DataLoader
import h5py
import torch as pt
from torch.nn import functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--output_path")
ifn = 'processed.hdf5'
ofn = args.output_path
chk = 'model.sav15'
batchsize= 20

a27 = ' ARNDCQEGHILKMFPSTWYVXUZJOB _'

vocab = {'A':1, 'R':2, 'N':3, 'D':4, 'C':5, 'Q':6, 'E':7, 'G':8, 'H':9, 'I':10,
        'L':11, 'K':12, 'M':13, 'F':14, 'P':15, 'S':16, 'T':17, 'W':18, 'Y':19,
        'V':20, 'X':21, 'U':22, 'Z':23, 'J':24, 'O':25, 'B':26, ' ':27, '_':28}
with h5py.File(ifn, 'r') as f:
    idx, seq = f['idx'][()], f['seq'][()]
sampler = SeqSampler(idx)
dataset = SeqDataset(seq, idx[-1])
dataloader = DataLoader(dataset=dataset, sampler=sampler, batch_size=batchsize, shuffle=False)
print('build instances')

model = Transformer(width=256, dimhead=64, numhead=8, depth=12).cuda()
model.load_state_dict(pt.load(chk)['model'])
print('build model')

def ret(x):
    return a27[x]

output = []
fragment_list = []
index_list = []
for input, index, pos in dataloader:
    input, pos = input.cuda(), pos.cuda()
    _, ylst = model(input, pos)
    p = ylst[-1]
    p = F.softmax(p, dim=-1)
    p = pt.argmax(p, dim=-1)
    p = p.cpu()
    p = p.numpy()
    input_pad = input.cpu()
    p[input_pad==0] = 0
    for i in range(len(p)):
        seq_string = ''
        for j in range(len(p[i])):
            if p[i][j] != 0:
                seq_string = seq_string + ret(p[i][j])
        fragment_list.append(seq_string)
        index_list.append(int(index[i]))
carrier = ''
for i in range(len(index_list)):
    carrier = carrier + fragment_list[i]
    if i != (len(index_list) - 1) and index_list[i] != index_list[i + 1]:
        output.append(carrier)
        carrier = ''
    if i == (len(index_list) - 1):
        output.append(carrier)

def seq_extracter(ifn):
    seq_list = []
    with open(ifn, 'r') as f:
        while True:
            line = f.readline()
            if line != '' and line[0] != '>':
                seq_list.append(line)
            if not line:
                break
        f.close()
    return seq_list

def checker(mask_path, output):
    masked_seq = seq_extracter(mask_path)
    if len(masked_seq) == len(output):
        print('output same number of sequences')
    for i in range(len(output)):
        error = 0
        if len(masked_seq[i]) != len(output[i]):
            error = error + 1
    print(f'the number of sequences that have different length is {error}')
mask_path = 'mounted_path/test.fasta'
#checker(mask_path, output)

def output_seq(ofn, output):
    with open(ofn, 'w') as f:
        for i in range(len(output)):
            f.write(output[i] + '\n')
        f.close()

output_seq(ofn, output)
