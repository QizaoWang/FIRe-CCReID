"""
@author:  Qizao Wang
@contact: qzwang22@m.fudan.edu.cn

TIFS 2024 paper: Exploring Fine-Grained Representation and Recomposition for Cloth-Changing Person Re-Identification
URL: https://ieeexplore.ieee.org/document/10557733
GitHub: https://github.com/QizaoWang/FIRe-CCReID
"""

import collections
import time
from tqdm import tqdm, trange
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import faiss

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data_process import samplers, transform
from data_process.dataset_loader_cc import ImageClothDataset_cc
from data_process.dataset_loader import ImageDataset
from model import fire
from scheduler.warm_up_multi_step_lr import WarmupMultiStepLR
from utils.util import AverageMeter
from utils.faiss_utils import search_index_pytorch, search_raw_array_pytorch, index_init_gpu, index_init_cpu


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]


def compute_jaccard_distance(target_features, k1=20, k2=6, print_flag=True, search_option=0, use_float16=False):
    end = time.time()
    if print_flag:
        print('Computing jaccard distance...')

    ngpus = faiss.get_num_gpus()
    N = target_features.size(0)
    mat_type = np.float16 if use_float16 else np.float32

    if (search_option==0):
        # GPU + PyTorch CUDA Tensors (1)
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        _, initial_rank = search_raw_array_pytorch(res, target_features, target_features, k1)
        initial_rank = initial_rank.cpu().numpy()
    elif (search_option==1):
        # GPU + PyTorch CUDA Tensors (2)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = search_index_pytorch(index, target_features, k1)
        res.syncDefaultStreamCurrentDevice()
        initial_rank = initial_rank.cpu().numpy()
    elif (search_option==2):
        # GPU
        index = index_init_gpu(ngpus, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)
    else:
        # CPU
        index = index_init_cpu(target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)


    nn_k1 = []
    nn_k1_half = []
    for i in range(N):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1/2))))

    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if (len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index)) > 2/3*len(candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        dist = 2-2*torch.mm(target_features[i].unsqueeze(0).contiguous(), target_features[k_reciprocal_expansion_index].t())
        if use_float16:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
        else:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()

    del nn_k1, nn_k1_half

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=mat_type)
        for i in range(N):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:], axis=0)
        V = V_qe
        del V_qe

    del initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:,i] != 0)[0])  #len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)
        # temp_max = np.zeros((1,N), dtype=mat_type)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]]+np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
            # temp_max[0,indImages[j]] = temp_max[0,indImages[j]]+np.maximum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])

        jaccard_dist[i] = 1-temp_min/(2-temp_min)
        # jaccard_dist[i] = 1-temp_min/(temp_max+1e-6)

    del invIndex, V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    if print_flag:
        print("Jaccard distance computing time cost: {}".format(time.time()-end))

    return jaccard_dist


def init_fg_cluster(args, model, dataset):
    transform_train, transform_test = transform.get_transform(args)

    if args.dataset in ['ltcc', 'prcc', 'deepchange', 'last']:
        train_loader_normal = DataLoader(
            ImageClothDataset_cc(dataset.train, transform=transform_test),
            batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers,
            pin_memory=True, drop_last=False,
        )
    else:
        train_loader_normal = DataLoader(
            ImageDataset(dataset.train, transform=transform_test),
            batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers,
            pin_memory=True, drop_last=False,
        )

    model.eval()
    dataid_dict = collections.defaultdict(list)
    feat_dict = collections.defaultdict(list)
    with torch.no_grad():
        dataid = 0
        for data in tqdm(train_loader_normal):
            if args.dataset in ['ltcc', 'prcc', 'deepchange', 'last']:
                img, pid, _, camid = data
            else:
                img, pid, camid = data

            feat = model(img.cuda())
            for i in range(img.shape[0]):
                dataid_dict[int(pid[i])].append(dataid)
                dataid += 1
                feat_dict[int(pid[i])].append(feat[i].unsqueeze(0))
    model.train()

    with torch.no_grad():
        num_pids = len(feat_dict.keys())
        num_fg_class_list = []
        dataid2fgid = collections.defaultdict(int)
        fg_center = []
        pseudo_train_dataset = []
        for pid in trange(num_pids):
            fg_feats = F.normalize(torch.concat(feat_dict[pid], dim=0), p=2, dim=1)
            dist_mat = compute_jaccard_distance(fg_feats, k1=args.k1, k2=args.k2, print_flag=False)
            cluster = DBSCAN(eps=args.eps, min_samples=1, metric='precomputed', n_jobs=-1)
            tmp_pseudo_fgids = cluster.fit_predict(dist_mat)

            # assign labels to outliers
            num_fgids = len(set(tmp_pseudo_fgids)) - (1 if -1 in tmp_pseudo_fgids else 0)
            pseudo_fgids = []
            for id in tmp_pseudo_fgids:
                if id != -1:
                    pseudo_fgids.append(id)
                else:  # outlier
                    pseudo_fgids.append(num_fgids)
                    num_fgids += 1

            def generate_cluster_features(labels, feats):
                feat_centers = collections.defaultdict(list)
                for i, label in enumerate(labels):
                    feat_centers[labels[i]].append(feats[i])
                feat_centers = [torch.stack(feat_centers[fgid], dim=0).mean(0).detach()
                                for fgid in sorted(feat_centers.keys())]  # n_fg [d]
                return torch.stack(feat_centers, dim=0)

            fg_centers = generate_cluster_features(pseudo_fgids, fg_feats)  # [n_fg, d]

            for i in range(len(dataid_dict[pid])):
                dataid2fgid[dataid_dict[pid][i]] = sum(num_fg_class_list) + pseudo_fgids[i]
            num_fg_class_list.append(num_fgids)
            fg_center.append(fg_centers)
            del fg_feats

        # get new train_loader with fine-grained pseudo label
        for dataid, data in enumerate(dataset.train):
            if args.dataset in ['ltcc', 'prcc', 'deepchange', 'last']:
                img, pid, _, camid = data
            else:
                img, pid, camid = data

            pseudo_train_dataset.append((img, pid, dataid2fgid[dataid], camid))

        fg_center = torch.concat(fg_center, dim=0).detach()
        num_fg_classes = sum(num_fg_class_list)
        pid2fgids = np.zeros((num_pids, num_fg_classes))
        fg_cnt = 0
        for i in range(num_pids):
            pid2fgids[i, fg_cnt: fg_cnt + num_fg_class_list[i]] = 1
            fg_cnt += num_fg_class_list[i]
        pid2fgids = torch.from_numpy(pid2fgids)

        return fg_center, pseudo_train_dataset, pid2fgids, num_fg_classes


def train(args, epoch, dataset, train_loader, model, classifier,
          optimizer, scheduler, class_criterion, metric_criterion, FFM_criterion, use_gpu):
    fg_start_epoch = args.fg_start_epoch
    FAR_times = model.module.K_times
    FAR_weight = args.FAR_weight

    class_losses = AverageMeter()
    triplet_losses = AverageMeter()
    FFM_losses = AverageMeter()
    FAR_losses = AverageMeter()

    num_fg_classes = 0
    if epoch > fg_start_epoch:
        fg_center, pseudo_train_dataset, pid2fgids, num_fg_classes = \
            init_fg_cluster(args, model, dataset)

        # get dataloader with fine-grained pseudo labels
        transform_train, transform_test = transform.get_transform(args)
        sampler = samplers.RandomIdentitySampler_cc(pseudo_train_dataset, batch_size=args.train_batch,
                                                    num_instances=args.num_instances)
        train_loader = DataLoader(
            ImageClothDataset_cc(pseudo_train_dataset, transform=transform_train),
            sampler=sampler, batch_size=args.train_batch, num_workers=args.num_workers,
            pin_memory=True, drop_last=True,
        )

        fg_classifier = fire.FgClassifier(feature_dim=model.module.feature_dim,
                                          num_classes=num_fg_classes, init_center=fg_center.detach())
        del fg_center
        parameters = list(model.parameters()) + list(classifier.parameters()) + list(fg_classifier.parameters())
        fg_classifier = nn.DataParallel(fg_classifier).cuda()
        optimizer = torch.optim.Adam(params=[{'params': parameters, 'initial_lr': args.lr}],
                                     lr=args.lr, weight_decay=args.weight_decay)
        scheduler = WarmupMultiStepLR(optimizer, milestones=args.step_milestones, gamma=args.gamma,
                                      warmup_factor=args.warm_up_factor, last_epoch=epoch - 2)

    for batch_idx, data in enumerate(tqdm(train_loader)):
        if args.dataset in ['ltcc', 'prcc', 'deepchange', 'last'] or epoch > fg_start_epoch:
            img, pid, fgid, _ = data
            if use_gpu:
                fgid = fgid.cuda()
        else:
            img, pid, _ = data

        if use_gpu:
            img, pid = img.cuda(), pid.cuda()

        model.train()
        classifier.train()

        if epoch > fg_start_epoch:
            feat, y_FAR = model(img, fgid)
        else:
            feat = model(img)

        loss = 0
        y = classifier(feat)
        class_loss = class_criterion(y, pid)
        loss += class_loss

        if epoch > fg_start_epoch:
            triplet_loss = metric_criterion(feat, pid)
            loss += triplet_loss

            if use_gpu:
                pid2fgids = pid2fgids.cuda()
            fg_classifier.train()

            y_fg = fg_classifier(feat)
            pos_mask = pid2fgids[pid]
            FFM_loss = FFM_criterion(y_fg, fgid, pos_mask)
            loss += FFM_loss

            FAR_loss = class_criterion(y_FAR, pid.repeat(FAR_times)) * FAR_weight
            loss += FAR_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        class_losses.update(class_loss.item(), pid.size(0))
        if epoch > fg_start_epoch:
            triplet_losses.update(triplet_loss.item(), pid.size(0))
            FFM_losses.update(FFM_loss.item(), pid.size(0))
            FAR_losses.update(FAR_loss.item(), pid.size(0))

    if args.print_train_info_epoch_freq != -1 and epoch % args.print_train_info_epoch_freq == 0:
        print('Epoch{0} Cls:{cls_loss.avg:.4f} Tri:{triplet_loss.avg:.4f} '
              'FFM:{FFM_loss.avg:.4f} FAR:{FAR_loss.avg:.4f} n_fg:{num_fg_classes} '.format(
            epoch, cls_loss=class_losses, triplet_loss=triplet_losses,
            FFM_loss=FFM_losses, FAR_loss=FAR_losses, num_fg_classes=num_fg_classes))

    scheduler.step()
