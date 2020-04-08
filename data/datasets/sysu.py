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


class SYSU(BaseImageDataset):
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

    def __init__(self, root='/home/zxh/sysu/split', verbose=True, **kwargs):
        super(SYSU, self).__init__()    # 继承 BaseImageDataset
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_i = osp.join(self.dataset_dir, 'i_train')
        self.train_v = osp.join(self.dataset_dir, 'v_train')
        self.train_g = osp.join(self.dataset_dir, 'g_train')

        self._check_before_run()

        train = self._process_dir(self.train_i, self.train_v, self.train_g, relabel=True)  # return dataset[]

        if verbose:  # 设置运行的时候显示详细信息
            print("=> SYSU-MM01 loaded")

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

    def _process_dir(self, dir_path_i, dir_path_v, dir_path_g, relabel=False):

        ids = os.listdir(dir_path_v)

        pid_container_v = set()    # visible人物序号
        pid_container_i = set()
        pid_container_g = set()

        for id in ids:
            cams_v = os.listdir(osp.join(dir_path_v, id))   # cam 1,2,4,5
            cams_i = os.listdir(osp.join(dir_path_i, id))
            cams_g = os.listdir(osp.join(dir_path_g, id))
            for cam in cams_v:
                imgs = glob.glob(osp.join(dir_path_v, id, cam, '*.jpg')) #glob.glob()返回所有匹配的文件路径列表。它只有一个参数pathname，定义了文件路径匹配规则，这里可以是绝对路径，也可以是相对路径。
                for img in imgs:
                    pid = int(id)
                    pid_container_v.add(pid)
            for cam in cams_i:
                imgs = glob.glob(osp.join(dir_path_i, id, cam, '*.jpg'))
                for img in imgs:
                    pid = int(id)
                    pid_container_i.add(pid)
            for cam in cams_g:
                imgs = glob.glob(osp.join(dir_path_g, id, cam, '*.jpg'))
                for img in imgs:
                    pid = int(id)
                    pid_container_g.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container_i)}  # 自然序号： 人物序号

        dataset = [] # 分别是v,g,i的(img, pid, camid)

        for id in ids:
            cams_v = os.listdir(osp.join(dir_path_v, id))
            for cam in cams_v:
                imgs = glob.glob(osp.join(dir_path_v, id, cam, '*.jpg'))
                for img in imgs:
                    pid = int(id)
                    camid = int(cam[3])
                    assert 1 <= camid <= 6
                    camid -= 1
                    if relabel: pid = pid2label[pid]
                    dataset.append((img, pid, camid))

        for id in ids:
            cams_g = os.listdir(osp.join(dir_path_g, id))
            for cam in cams_g:
                imgs = glob.glob(osp.join(dir_path_g, id, cam, '*.jpg'))
                for img in imgs:
                    pid = int(id)
                    camid = int(cam[3])
                    assert 1 <= camid <= 6
                    camid -= 1
                    if relabel: pid = pid2label[pid]
                    dataset.append((img, pid, camid))

        for id in ids:
            cams_i = os.listdir(osp.join(dir_path_i, id))
            for cam in cams_i:
                imgs = glob.glob(osp.join(dir_path_i, id, cam, '*.jpg'))
                for img in imgs:
                    pid = int(id)
                    camid = int(cam[3])
                    assert 1 <= camid <= 6
                    camid -= 1
                    if relabel: pid = pid2label[pid]
                    dataset.append((img, pid, camid))

        return dataset
