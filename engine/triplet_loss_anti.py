# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
from torch import nn
import torch.nn.functional as F

def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]

    '================================================================================================'
    # dist_ap, relative_p_inds = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    dist_ap, relative_p_inds = torch.min(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    '================================================================================================'

    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]

    '================================================================================================'
    # dist_an, relative_n_inds = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    dist_an, relative_n_inds = torch.max(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    '================================================================================================'

    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels).copy_(torch.arange(0, N).long()).unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds
    return dist_ap, dist_an


class TripletLoss(object):
    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, fi, fv, labels):

        dist_mat = euclidean_dist(fi, fv)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            # loss = self.ranking_loss(dist_an, dist_ap, y)
            loss = self.ranking_loss(dist_ap, dist_an, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class CrossEntropyLabelSmooth_neg(nn.Module):
    def __init__(self, num_classes):
        super(CrossEntropyLabelSmooth_neg, self).__init__()
        self.num_classes = num_classes
        self.epsilon = 0.1
        self.use_gpu = True
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):

        # log_probs = self.logsoftmax(inputs)
        log_probs = torch.log(F.softmin(inputs, dim=1))


        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss