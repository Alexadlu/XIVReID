# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re
import os

import os.path as osp

from .bases import BaseImageDataset


class RegDB(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = ''

    def __init__(self, root='/data1/lidg/reid_dataset/IV-ReID/RegDB', verbose=True, **kwargs):
        super(RegDB, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_i = osp.join(self.dataset_dir, '')
        self.train_v = osp.join(self.dataset_dir, '')

        self._check_before_run()

        train = self._process_dir(self.train_i, self.train_v, relabel=True)

        if verbose:
            print("=> RegDB loaded")

        self.train = train

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_i):
            raise RuntimeError("'{}' is not available".format(self.train_i))
        if not osp.exists(self.train_v):
            raise RuntimeError("'{}' is not available".format(self.train_v))

    def _process_dir(self, dir_path_i, dir_path_v, relabel=False):

        with open('/data1/lidg/reid_dataset/IV-ReID/RegDB/idx/train_visible_0.txt','r') as f:
            files_v = f.readlines()
        with open('/data1/lidg/reid_dataset/IV-ReID/RegDB/idx/train_thermal_0.txt','r') as f:
            files_i = f.readlines()

        dataset = []

        for i in range(len(files_v)):
            img_v, id = files_v[i].strip().split(' ')
            img = osp.join(dir_path_v, img_v)
            pid = int(id)
            camid = 0
            dataset.append((img, pid, camid))

        for i in range(len(files_i)):
            img_i, id = files_i[i].strip().split(' ')
            img = osp.join(dir_path_i, img_i)
            pid = int(id)
            camid = 1
            dataset.append((img, pid, camid))

        return dataset
