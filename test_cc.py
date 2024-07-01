"""
@author:  Qizao Wang
@contact: qzwang22@m.fudan.edu.cn

TIFS 2024 paper: Exploring Fine-Grained Representation and Recomposition for Cloth-Changing Person Re-Identification
URL: https://ieeexplore.ieee.org/document/10557733
GitHub: https://github.com/QizaoWang/FIRe-CCReID
"""

from tqdm import tqdm
import numpy as np

import torch
from torch.nn import functional as F

from evaluate.metrics import evaluate
from test import get_distmat
from evaluate.metrics_for_cc import evaluate_ltcc, evaluate_prcc_all_gallery


def get_data_for_cc(datasetloader, use_gpu, model):
    with torch.no_grad():
        feats, pids, clothids, camids= [], [], [], []
        for batch_idx, (img, pid, clothid, camid) in enumerate(tqdm(datasetloader)):
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
            clothids.extend(clothid)
            camids.extend(camid)
        feats = torch.cat(feats, 0)
        pids = np.asarray(pids)
        clothids = np.asarray(clothids)
        camids = np.asarray(camids)
    return feats, pids, clothids, camids


def test_for_prcc(args, query_sc_loader, query_cc_loader, gallery_loader, model,
                  use_gpu, ranks=[1, 5, 10], epoch=None):
    model.eval()
    gf, g_pids, g_clothids, g_camids = get_data_for_cc(gallery_loader, use_gpu, model)
    qf, q_pids, q_clothids, q_camids = get_data_for_cc(query_sc_loader, use_gpu, model)
    distmat = get_distmat(qf, gf)
    cmc, mAP = evaluate_prcc_all_gallery(distmat, q_pids, g_pids)
    if epoch: print("Epoch {}: ".format(epoch), end='')
    print("mAP: {:.4%}  ".format(mAP), end='')
    for r in ranks:
        print("R-{:<2}: {:<7.4%}  ".format(r, cmc[r - 1]), end='')
    print()

    qf, q_pids, q_clothids, q_camids = get_data_for_cc(query_cc_loader, use_gpu, model)
    distmat = get_distmat(qf, gf)
    cmc_2, mAP_2 = evaluate_prcc_all_gallery(distmat, q_pids, g_pids)
    if epoch: print("Epoch {}: ".format(epoch), end='')
    print("mAP: {:.4%}  ".format(mAP_2), end='')
    for r in ranks:
        print("R-{:<2}: {:<7.4%}  ".format(r, cmc_2[r - 1]), end='')
    print()

    return [cmc[0], cmc_2[0]], [mAP, mAP_2]


def test_for_ltcc(args, query_loader, gallery_loader, model, use_gpu, ranks=[1, 5, 10], epoch=None):
    model.eval()
    qf, q_pids, q_clothids, q_camids = get_data_for_cc(query_loader, use_gpu, model)
    gf, g_pids, g_clothids, g_camids = get_data_for_cc(gallery_loader, use_gpu, model)
    distmat = get_distmat(qf, gf)

    cmc, mAP = evaluate_ltcc(distmat, q_pids, g_pids, q_camids, g_camids, q_clothids, g_clothids,
                             ltcc_cc_setting=False)
    if epoch: print("Epoch {}: ".format(epoch), end='')
    print("mAP: {:.4%}  ".format(mAP), end='')
    for r in ranks:
        print("R-{:<2}: {:<7.4%}  ".format(r, cmc[r - 1]), end='')
    print()

    cmc_2, mAP_2 = evaluate_ltcc(distmat, q_pids, g_pids, q_camids, g_camids, q_clothids, g_clothids,
                             ltcc_cc_setting=True)
    if epoch: print("Epoch {}: ".format(epoch), end='')
    print("mAP: {:.4%}  ".format(mAP_2), end='')
    for r in ranks:
        print("R-{:<2}: {:<7.4%}  ".format(r, cmc_2[r - 1]), end='')
    print()

    return [cmc[0], cmc_2[0]], [mAP, mAP_2]


def test_for_cc(args, query_loader, gallery_loader, model, use_gpu, ranks=[1, 5, 10], epoch=None):
    model.eval()
    qf, q_pids, q_clothids, q_camids = get_data_for_cc(query_loader, use_gpu, model)
    gf, g_pids, g_clothids, g_camids = get_data_for_cc(gallery_loader, use_gpu, model)
    distmat = get_distmat(qf, gf)

    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    if epoch: print("Epoch {}: ".format(epoch), end='')
    print("mAP: {:.4%}  ".format(mAP), end='')
    for r in ranks:
        print("R-{:<2}: {:<7.4%}  ".format(r, cmc[r - 1]), end='')
    print()

    return cmc[0], mAP
