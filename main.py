#this is my first version of training script
#this script should use Pytorch DistributedDataParallel to train a bert model using protein sequence
#out of this module, I have written model, sampler, dataset code.

import argparse
import os
import time
import math
import h5py
import torch.distributed as dist
import torch as pt
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim
from torch.utils.data import DataLoader
import wandb

from data import *
from model import *


lr = 5e-5
sched_cycle = 64
weight_decay = 1e-2
batchsize = 50
warmup_ratio = 1 / 120

width = 256
dimhead = 64
numhead = 8
depth = 12

chk= '/home/Zhaoxu/Project/output/ep8-tune/model.sav15'

def dist_setting():
    """make sure DDP work well"""
    dist.init_process_group(backend="nccl")
    #this ensure each process works on a single GPU
    pt.cuda.set_device(rank)


def param_group():
    """create parameter group"""
    pgrp0, pgrp1 = [], []
    for n, p in ddp_model.named_parameters():
        if n.endswith('weight'): pgrp1.append(p)
        else: pgrp0.append(p)
    pgrp = [{'params': pgrp0, 'weight_decay': 0}, {'params': pgrp1, 'weight_decay': weight_decay}]
    return pgrp

 
def evaluate(plst, gt, mask):
    with pt.no_grad():
        mm = (gt > 0).type_as(plst[-1])
        m0, m1 = mm * (1 - mask), mm * mask
        m0 /= pt.sum(m0, dim=-1, keepdim=True) + 1e-4
        m1 /= pt.sum(m1, dim=-1, keepdim=True) + 1e-4
    l0, l1 = 0, 0
    for i, p in enumerate(plst):
        p, gt = p.reshape(-1, p.shape[-1]), gt.reshape(-1)
        pp = F.softmax(p, dim=-1)[pt.arange(len(gt)), gt]
        ll = (1 - pp).pow(2.0) * F.cross_entropy(p, gt, reduction='none')
        l0 = l0 + pt.sum(ll.reshape(m0.shape) * m0) / m0.shape[0]
        l1 = l1 + pt.sum(ll.reshape(m1.shape) * m1) / m1.shape[0]
    l0, l1 = l0 / len(plst), l1 / len(plst)
    with pt.no_grad():
        aa = (pt.argmax(p, dim=-1) == gt).type_as(p) * 100
        a0 = pt.sum(aa.reshape(m0.shape) * m0) / m0.shape[0]
        a1 = pt.sum(aa.reshape(m1.shape) * m1) / m1.shape[0]
    return l0, l1, a0, a1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local_world_size", type=int, default=1)
    parser.add_argument("--expe_name")
    args = parser.parse_args()

    rank, world = int(args.local_rank), int(args.local_world_size)
    expe_name = args.expe_name
    dist_setting()

    print('loading data ...')
    with h5py.File(ifn, 'r') as f:
        idx, seq = f['idx'][()], f['seq'][()]
    sampler = SeqSampler(idx, rank=rank, world=world)
    dataset = SeqDataset(seq, idx[-1])
    dataloader = DataLoader(dataset=dataset, sampler=sampler, num_workers=2, prefetch_factor=batchsize)

    print('building model ...')
    model = Transformer(width, dimhead, numhead, depth).cuda()
    if rank == 0:
        try: model.load_state_dict(pt.load(chk)['model'])
        except: pass
    ddp_model = DDP(model, device_ids=[rank])

    pgrp = param_group()
    optimizer = optim.AdamW(pgrp, lr)

    def lr_lambda(current_step: int):
        num_training_steps = 1500_000
        num_warmup_steps = float(warmup_ratio * num_training_steps)
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    # if rank == 0:
    #     optimizer.load_state_dict(pt.load(chk)['optimizer'])
    #     scheduler.load_state_dict(pt.load(chk)['scheduler'])

    schedsize = 2 * len(sampler) // sched_cycle // batchsize // world
    epochsize = schedsize * sched_cycle // 2

    ddp_model.train()
    config = {
        "lr": lr,
        "batchsize": batchsize,
        "epochsize": epochsize,
        "schedsize": schedsize,
        "width": width,
        "dimhead": dimhead,
        "numhead": numhead,
        "depth": depth}

    batch = 0
    if rank == 0:
        print();
        print('#training model ...')
        stat, savfn = [], f'output/{expe_name}/model.sav'
        tchk = tsched = time.perf_counter()
        wandb.init(project='protein_bert', entity='zhaoxu',
                   config=config)
        wandb.watch(model)

    for x, y, mask, pos in dataloader:
        with pt.no_grad():
            if batch % schedsize == 0:
                sched = batch // schedsize
                if rank == 0:
                    tnow = time.perf_counter()
                    pt.save({'prog': batch * batchsize * world, 'sched': sched, 'random': random.getstate(),
                             'model': model.state_dict(),
                             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()},
                            savfn + str(sched))
                    print('#sched[%d]: %.2e %.1fm' % (sched, optimizer.param_groups[0]['lr'], (tnow - tsched) / 60))
                    tsched = tnow

        optimizer.zero_grad()
        _, ylst = ddp_model(x, pos)
        loss0, loss1, acc0, acc1 = evaluate(ylst, y.cuda(), mask.cuda())
        loss = loss1 + loss0 / 100
        # loss = loss1
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()
        batch += 1

        if rank > 0: continue
        with pt.no_grad():
            tnow = time.perf_counter()
            stat.append([loss0.item(), loss1.item(), acc0.item(), acc1.item()])
            if tnow - tchk > 59:
            #if batch % epochsize == 0:
                stat = list(np.mean(stat, axis=0))
                print('#prog[%.6f]: %.4f %.4f %.2f%% %.2f%% %.1fs' % (batch/epochsize, *stat, tnow-tchk))
                #print('#prog[%.1f]: %.4f %.4f %.2f%% %.2f%% %.1fs' % (batch / epochsize, *stat, tnow - tchk))
                wandb.log({"loss0": stat[0], "loss1": stat[1], "acco": stat[2], "acc1": stat[3], 
                    "epoch": batch//epochsize +1, "current_lr": optimizer.param_groups[0]['lr'], "batch": batch})
                stat, tchk = [], tnow

    # Mark the run as finished
    wandb.finish()
    dist.destroy_process_group()



