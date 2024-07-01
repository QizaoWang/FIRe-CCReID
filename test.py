"""
@author:  Qizao Wang
@contact: qzwang22@m.fudan.edu.cn

TIFS 2024 paper: Exploring Fine-Grained Representation and Recomposition for Cloth-Changing Person Re-Identification
URL: https://ieeexplore.ieee.org/document/10557733
GitHub: https://github.com/QizaoWang/FIRe-CCReID
"""

from __future__ import absolute_import

from tqdm import tqdm, trange
import numpy as np

import torch
from torch.nn import functional as F

from evaluate.metrics import evaluate
from evaluate.re_ranking import re_ranking


def get_data(datasetloader, use_gpu, model):
    with torch.no_grad():
        feats, pids, camids= [], [], []
        for batch_idx, (img, pid, camid) in enumerate(tqdm(datasetloader)):
            flip_img = torch.flip(img, [3])
            if use_gpu:
                img, flip_img = img.cuda(), flip_img.cuda()
            feat = model(img)
            feat_flip = model(flip_img)
            feat += feat_flip
            feat = F.normalize(feat, p=2, dim=1)
            feat = feat.data.cpu()
            feats.append(feat)
            pids.extend(pid)
            camids.extend(camid)
        feats = torch.cat(feats, 0)
        pids = np.asarray(pids)
        camids = np.asarray(camids)
    return feats, pids, camids


def get_distmat(qf, gf):
    # feature normalization
    qf = 1. * qf / (torch.norm(qf, 2, dim=-1, keepdim=True).expand_as(qf) + 1e-12)
    gf = 1. * gf / (torch.norm(gf, 2, dim=-1, keepdim=True).expand_as(gf) + 1e-12)
    # get distmat matrix
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    distmat = distmat.numpy()
    return distmat


def test(args, queryloader, galleryloader, model, use_gpu, ranks=[1, 5, 10], epoch=None):
    model.eval()
    with torch.no_grad():
        qf, q_pids, q_camids = get_data(queryloader, use_gpu, model)
        gf, g_pids, g_camids = get_data(galleryloader, use_gpu, model)

    distmat = get_distmat(qf, gf)

    if args.reranking:
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        print("With Reranking: ", end='')

    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    if epoch: print("Epoch {}: ".format(epoch), end='')
    print("mAP: {:.4%}  ".format(mAP), end='')
    for r in ranks:
        print("Rank-{:<2}: {:<7.4%}  ".format(r, cmc[r - 1]), end='')
    print()

    return cmc[0], mAP

