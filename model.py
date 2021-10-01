#!/usr/bin/env -S python3 -Bu
# add gate

import numpy as np
import torch as pt

from torch import nn
from torch.nn import functional as F
#from performer_pytorch import FastAttention




dimgroup, dimwrap = 16, 32

class ReZero(nn.Module):
    def __init__(self, cio):
        super(ReZero, self).__init__()
        self.res = nn.parameter.Parameter(pt.zeros([cio//dimgroup, 1]), requires_grad=True)

    def forward(self, x):  # N x C x L
        shape = [x.shape[0], x.shape[1]//dimgroup, -1]
        xx = (x.reshape(shape) * self.res).reshape(x.shape)
        return xx

class GateZero(nn.Module):
    def __init__(self, cio):
        super(GateZero, self).__init__()
        self.res = nn.parameter.Parameter(pt.zeros([cio//dimgroup, 1, 1]), requires_grad=True)
        self.gate = nn.Sequential(nn.Conv1d(cio, cio//dimgroup, kernel_size=1), nn.Sigmoid())

    def forward(self, x, v):  # N x C x L
        shape = [x.shape[0], x.shape[1]//dimgroup, dimgroup, x.shape[2]]
        xx = (v.reshape(shape) * self.gate(x)[:, :, None, :] * self.res).reshape(v.shape)
        return xx

class PosEmbed(nn.Module):
    def __init__(self, cio, maxembed=32*12):
        super(PosEmbed, self).__init__()
        self.maxembed = maxembed

        self.embed0 = nn.Embedding(maxembed+1, cio//2, padding_idx=0, max_norm=2.0)
        self.embed1 = nn.Embedding(maxembed+1, cio//2, padding_idx=0, max_norm=2.0)

    def forward(self, x, pos):  # N x L x C
        x0 = pos[:, 0][:, None] + pt.arange(x.shape[1], device='cuda')[None, :]
        x0[x0 > self.maxembed] = self.maxembed
        x1 = pos[:, 1][:, None] + pt.arange(x.shape[1], device='cuda')[None, :]
        x1[x1 > self.maxembed] = self.maxembed
        xx = pt.cat([self.embed0(x0), self.embed1(x1)], dim=-1)
        return xx

class AttBlock(nn.Module):
    def __init__(self, cio, dimhead, numhead, maxembed=32*12):
        super(AttBlock, self).__init__()
        self.cio = cio
        self.chid = dimhead * numhead
        self.dimhead = dimhead
        self.numhead = numhead
        self.maxembed = maxembed

        self.qkpos0 = ReZero(cio)
        self.qkorg0 = ReZero(cio)
        self.vorg0 = ReZero(cio)
        self.relpos = nn.parameter.Parameter(pt.zeros(self.maxembed*2-1), requires_grad=True)
        self.gate = nn.Conv1d(cio, numhead, kernel_size=1)

        self.q = nn.Conv1d(cio, self.chid, kernel_size=1, bias=False)
        self.k = nn.Conv1d(cio, self.chid, kernel_size=1, bias=False)
        self.v = nn.Conv1d(cio, self.chid, kernel_size=1, bias=False)
        self.dense = nn.Conv1d(self.chid, cio, kernel_size=1)

    def forward(self, x, xorg, abspos, mask, norm):  # N x C x L
        N, C, L = x.shape
        x0, x1 = x + self.vorg0(xorg), x + self.qkorg0(xorg) + self.qkpos0(abspos)

        # dot product
        xx, x1 = pt.bmm(self.q(x1).reshape(-1, self.dimhead, L).permute(0, 2, 1), \
                self.k(x1).reshape(-1, self.dimhead, L)), None
        # relative position, gate, mask, norm
        idx = pt.arange(self.maxembed, self.maxembed-L, -1)[:, None] + pt.arange(L)[None, :]
        idx[idx<0] = 0; idx[idx>=len(self.relpos)] = len(self.relpos) - 1
        xx, idx = (xx.reshape(N, self.numhead, L, L) + self.relpos[idx]
                + self.gate(x)[:, :, None, :] + mask[:, None, None, :]) \
                / norm[:, None, None, None], None
        # softmax sum
        xx, x0 = pt.bmm(F.softmax(xx, dim=-1).reshape(-1, L, L), \
                self.v(x0).reshape(-1, self.dimhead, L).permute(0, 2, 1)), None
        # dense
        xx = self.dense(xx.permute(0, 2, 1).reshape(N, -1, L))

        return xx

class DenseBlock(nn.Module):
    def __init__(self, cio, dimhead, numhead):
        super(DenseBlock, self).__init__()
        self.cio = cio
        self.chid = dimhead * numhead
        self.dimhead = dimhead
        self.numhead = numhead

        self.xorg0 = ReZero(cio)
        self.dense = nn.Sequential(nn.Conv1d(cio, self.chid, kernel_size=1, groups=1),
                nn.GELU(), nn.Conv1d(self.chid, self.chid, kernel_size=1, groups=numhead),
                nn.GELU(), nn.Conv1d(self.chid, cio, kernel_size=1, groups=1))

    def forward(self, x, xorg):  # N x C x L
        return self.dense(x + self.xorg0(xorg))


class Transformer(nn.Module):
    def __init__(self, width, dimhead, numhead, depth):
        super(Transformer, self).__init__()
        self.width = width
        self.dimhead = dimhead
        self.numhead = numhead
        self.depth = depth

        self.embed = nn.Embedding(28+1, width, padding_idx=0, max_norm=2.0)
        self.abspos = PosEmbed(width)
        self.norm = nn.GroupNorm(1, width, affine=False)
        self.body = nn.ModuleList()
        for i in range(self.depth):
            self.body.append(nn.ModuleList([
                    AttBlock(width, dimhead, numhead), GateZero(width),
                    DenseBlock(width, dimhead*2, numhead), GateZero(width),
                    nn.Linear(width, 28)]))

    def forward(self, x, pos):  # N x L x C
        xlst, ylst = [], []
        x0 = xx = self.embed(x).permute(0, 2, 1)  # N x C x L
        xp = self.abspos(x, pos).permute(0, 2, 1)  # N x C x L
        idx = pt.arange(x.shape[1], dtype=pos.dtype, device='cuda')[None, :] >= pos[:, -1][:, None]
        xm = pt.zeros(idx.shape, dtype=xx.dtype, device='cuda'); xm[idx] = float('-inf')  # N x L
        xn = pt.sqrt(pos[:, -1].type_as(x))  # N
        for att, att0, dense, dense0, head in self.body:
            xx = self.norm(xx + att0(xx, att(xx, x0, xp, xm, xn)))
            xx = self.norm(xx + dense0(xx, dense(xx, x0)))
            yy = head(xx.permute(0, 2, 1))   # N x L x C
            xlst.append(xx)
            ylst.append(yy)
        return xlst, ylst

