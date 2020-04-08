# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import copy
import random
import torch
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        # len(self.data_source)=12936  [('/home/.../.jpg', id, cam), (), ...]
        # SYSU  visible:22258  +  infrared:11909
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # len(self.index_dic) = 751

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances


    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)

        return iter(final_idxs)

    def __len__(self):
        return self.length




class RandomIdentitySampler_SYSU(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):


        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances  #48/4

        self.index_dic1 = defaultdict(list)
        self.index_dic2 = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            if index<22258:
                self.index_dic1[pid].append(index)
            #else:
            elif index>=(22258*2):
                self.index_dic2[pid].append(index)

        self.pids = list(self.index_dic1.keys())

        # len(self.index_dic) = 751

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs1 = self.index_dic1[pid]
            idxs2 = self.index_dic2[pid]
            num1 = len(idxs1)
            num2 = len(idxs2)
            num = max(num1, num2)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances


    def __iter__(self):
        batch_idxs_dict1 = defaultdict(list)
        batch_idxs_dict2 = defaultdict(list)

        for pid in self.pids:
            idxs1 = copy.deepcopy(self.index_dic1[pid])
            idxs2 = copy.deepcopy(self.index_dic2[pid])

            num1 = len(idxs1)
            num2 = len(idxs2)
            num = max(num1, num2)

            if num < self.num_instances:
                idxs1 = np.random.choice(idxs1, size=self.num_instances, replace=True).tolist()
                idxs2 = np.random.choice(idxs2, size=self.num_instances, replace=True).tolist()

            if num==num1:
                new_idxs2 = []
                need = num-num2
                for ii in range(need//num2 + 1):
                    new_idxs2.extend(idxs2)
                extra = np.random.choice(idxs2, need%num2, replace=False).tolist()
                new_idxs2.extend(extra)
                idxs2 = new_idxs2
            else:
                new_idxs1 = []
                need = num-num1
                for ii in range(need//num1 + 1):
                    new_idxs1.extend(idxs1)
                extra = np.random.choice(idxs1, need%num1, replace=False).tolist()
                new_idxs1.extend(extra)
                idxs1 = new_idxs1

            random.shuffle(idxs1)
            random.shuffle(idxs2)
            batch_idxs1 = []
            batch_idxs2 = []
            for idx1, idx2 in zip(idxs1, idxs2):
                batch_idxs1.append(idx1)
                batch_idxs2.append(idx2)
                if len(batch_idxs1) == self.num_instances:
                    batch_idxs_dict1[pid].append(batch_idxs1)
                    batch_idxs_dict2[pid].append(batch_idxs2)
                    batch_idxs1 = []
                    batch_idxs2 = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs1 = batch_idxs_dict1[pid].pop(0)
                batch_idxs2 = batch_idxs_dict2[pid].pop(0)

                batch_idxs = []
                for i in range(4):
                    batch_idxs.append(batch_idxs1[i])
                    batch_idxs.append(batch_idxs2[i])

                #batch_idxs = batch_idxs1 + batch_idxs2

                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict1[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomIdentitySampler_SYSU_Thr(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):

        # len(self.data_source)=12936  [('/home/.../.jpg', id, cam), (), ...]
        # SYSU  visible:22258  +  infrared:11909

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances

        self.index_dic1 = defaultdict(list)
        self.index_dic2 = defaultdict(list)
        self.index_dic3 = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            if index<22258:
                self.index_dic1[pid].append(index)
            elif index<22258*2:
                self.index_dic3[pid].append(index)
            else:
                self.index_dic2[pid].append(index)

        self.pids = list(self.index_dic1.keys())

        # len(self.index_dic) = 751

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs1 = self.index_dic1[pid]
            idxs2 = self.index_dic2[pid]
            num1 = len(idxs1)
            num2 = len(idxs2)
            num = max(num1, num2)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances


    def __iter__(self):
        batch_idxs_dict1 = defaultdict(list)
        batch_idxs_dict2 = defaultdict(list)
        batch_idxs_dict3 = defaultdict(list)

        for pid in self.pids:
            idxs1 = copy.deepcopy(self.index_dic1[pid])
            idxs2 = copy.deepcopy(self.index_dic2[pid])
            idxs3 = copy.deepcopy(self.index_dic3[pid])

            num1 = len(idxs1)
            num2 = len(idxs2)
            num = max(num1, num2)

            if num < self.num_instances:
                idxs1 = np.random.choice(idxs1, size=self.num_instances, replace=True).tolist()
                idxs2 = np.random.choice(idxs2, size=self.num_instances, replace=True).tolist()
                idxs3 = np.random.choice(idxs3, size=self.num_instances, replace=True).tolist()

            if num==num1:
                new_idxs2 = []
                need = num-num2
                for ii in range(need//num2 + 1):
                    new_idxs2.extend(idxs2)
                extra = np.random.choice(idxs2, need%num2, replace=False).tolist()
                new_idxs2.extend(extra)
                idxs2 = new_idxs2
            else:
                new_idxs1 = []
                new_idxs3 = []
                need = num-num1
                for ii in range(need//num1 + 1):
                    new_idxs1.extend(idxs1)
                    new_idxs3.extend(idxs3)
                extra1 = np.random.choice(idxs1, need%num1, replace=False).tolist()
                extra3 = np.random.choice(idxs3, need%num1, replace=False).tolist()
                new_idxs1.extend(extra1)
                new_idxs3.extend(extra3)
                idxs1 = new_idxs1
                idxs3 = new_idxs3

            random.shuffle(idxs1)
            random.shuffle(idxs2)
            random.shuffle(idxs3)
            batch_idxs1 = []
            batch_idxs2 = []
            batch_idxs3 = []
            for idx1, idx2, idx3 in zip(idxs1, idxs2, idxs3):
                batch_idxs1.append(idx1)
                batch_idxs2.append(idx2)
                batch_idxs3.append(idx3)
                if len(batch_idxs1) == self.num_instances:
                    batch_idxs_dict1[pid].append(batch_idxs1)
                    batch_idxs_dict2[pid].append(batch_idxs2)
                    batch_idxs_dict3[pid].append(batch_idxs3)
                    batch_idxs1 = []
                    batch_idxs2 = []
                    batch_idxs3 = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs1 = batch_idxs_dict1[pid].pop(0)
                batch_idxs2 = batch_idxs_dict2[pid].pop(0)
                batch_idxs3 = batch_idxs_dict3[pid].pop(0)

                batch_idxs = []
                for i in range(4):
                    batch_idxs.append(batch_idxs1[i])
                    batch_idxs.append(batch_idxs2[i])
                    batch_idxs.append(batch_idxs3[i])
                #batch_idxs = batch_idxs1 + batch_idxs2 + batch_idxs3

                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict1[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomIdentitySampler_RegDB(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):

        # len(self.data_source)=12936  [('/home/.../.jpg', id, cam), (), ...]
        # SYSU  visible:22258  +  infrared:11909

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances

        self.index_dic1 = defaultdict(list)
        self.index_dic2 = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            if index<2060*2:
                self.index_dic1[pid].append(index)
            else:
                self.index_dic2[pid].append(index)

        self.pids = list(self.index_dic1.keys())

        # len(self.index_dic) = 751

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs1 = self.index_dic1[pid]
            idxs2 = self.index_dic2[pid]
            num1 = len(idxs1)
            num2 = len(idxs2)
            num = max(num1, num2)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances


    def __iter__(self):
        batch_idxs_dict1 = defaultdict(list)
        batch_idxs_dict2 = defaultdict(list)

        for pid in self.pids:
            idxs1 = copy.deepcopy(self.index_dic1[pid])
            idxs2 = copy.deepcopy(self.index_dic2[pid])

            random.shuffle(idxs1)
            random.shuffle(idxs2)
            batch_idxs1 = []
            batch_idxs2 = []
            for idx1, idx2 in zip(idxs1, idxs2):
                batch_idxs1.append(idx1)
                batch_idxs2.append(idx2)
                if len(batch_idxs1) == self.num_instances:
                    batch_idxs_dict1[pid].append(batch_idxs1)
                    batch_idxs_dict2[pid].append(batch_idxs2)
                    batch_idxs1 = []
                    batch_idxs2 = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs1 = batch_idxs_dict1[pid].pop(0)
                batch_idxs2 = batch_idxs_dict2[pid].pop(0)
                batch_idxs = batch_idxs1 + batch_idxs2
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict1[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)

        return iter(final_idxs)

    def __len__(self):
        return self.length


# New add by gu
class RandomIdentitySampler_alignedreid(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances
