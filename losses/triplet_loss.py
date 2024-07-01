from __future__ import absolute_import

import torch
from torch import nn
import torch.nn.functional as F
from utils.util import euclidean_dist


def hard_example_mining(dist_mat, labels, mining_method='batch_hard', return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    if mining_method == 'batch_hard':
        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap, relative_p_inds = torch.max(
            dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
        # `dist_an` means distance(anchor, negative)
        # both `dist_an` and `relative_n_inds` with shape [N, 1]
        dist_an, relative_n_inds = torch.min(
            dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
        # shape [N]
    elif mining_method == 'batch_sample':
        dist_mat_ap = dist_mat[is_pos].contiguous().view(N, -1)
        relative_p_inds = torch.multinomial(
            F.softmax(dist_mat_ap, dim=1), num_samples=1)
        dist_ap = torch.gather(dist_mat_ap, 1, relative_p_inds)

        dist_mat_an = dist_mat[is_neg].contiguous().view(N, -1)
        relative_n_inds = torch.multinomial(
            F.softmin(dist_mat_an, dim=1), num_samples=1)
        dist_an = torch.gather(dist_mat_an, 1, relative_n_inds)
    elif mining_method == 'batch_soft':
        dist_mat_ap = dist_mat[is_pos].contiguous().view(N, -1)
        dist_mat_an = dist_mat[is_neg].contiguous().view(N, -1)
        weight_ap = torch.exp(dist_mat_ap) / torch.exp(dist_mat_ap).sum(dim=1, keepdim=True)
        weight_an = torch.exp(-dist_mat_an) / torch.exp(-dist_mat_an).sum(dim=1, keepdim=True)

        dist_ap = (weight_ap * dist_mat_ap).sum(dim=1, keepdim=True)
        dist_an = (weight_an * dist_mat_an).sum(dim=1, keepdim=True)
    else:
        print("error, unsupported mining method {}".format(mining_method))

    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None, mining_method='batch_hard'):
        self.margin = margin
        self.mining_method = mining_method
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, features, labels):
        dist_mat = euclidean_dist(features, features)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels, self.mining_method)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss
