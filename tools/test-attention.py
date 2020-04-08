# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import torch
from torch.backends import cudnn
import torchvision.transforms as transforms
import torch.utils.data as data

sys.path.append('.')
from config import cfg
from data import make_data_loader
from modeling import build_model
from utils.logger import setup_logger

import logging, copy

import torch.nn as nn
from ignite.engine import Engine

from utils.reid_metric import R1_mAP, R1_mAP_reranking
import time

from utils.data_loader import TestData
from utils.data_manager import *
from utils.eval_metrics import eval_sysu
from utils.utils import *
from utils.ecn import ECN

torch.cuda.manual_seed_all(0)
torch.manual_seed(0)
np.random.seed(0)

def create_supervised_evaluator(model, metrics, device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
def inference(cfg,model,val_loader,num_query):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")
    if cfg.TEST.RE_RANKING == 'no':
        print("Create evaluator")
        evaluator = create_supervised_evaluator(
            model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    elif cfg.TEST.RE_RANKING == 'yes':
        print("Create evaluator for reranking")
        evaluator = create_supervised_evaluator(
            model, metrics={'r1_mAP': R1_mAP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    else:
        print("Unsupported re_ranking config. Only support for no or yes, but got {}.".format(cfg.TEST.RE_RANKING))

    evaluator.run(val_loader)
    cmc, mAP = evaluator.state.metrics['r1_mAP']
    logger.info('Validation Results')
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
def euclidean_dist(anchor, positive):

    # d1 = torch.sum(anchor * anchor, dim=1).unsqueeze(-1)
    # d2 = torch.sum(positive * positive, dim=1).unsqueeze(-1)
    # eps = 1e-12
    # a = d1.repeat(1, positive.size(0))
    # b = torch.t(d2.repeat(1, anchor.size(0)))
    # c = 2.0 * torch.bmm(anchor.unsqueeze(0), torch.t(positive).unsqueeze(0)).squeeze(0)
    # return torch.sqrt(torch.abs((a + b - c)) + eps)

    m, n = anchor.shape[0], positive.shape[0]
    distmat = torch.pow(anchor, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(positive, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, anchor, positive.t())
    return distmat.numpy()
def norm(x):
    x = 1. * x / (torch.norm(x, 2, -1, keepdim=True).expand_as(x) + 1e-12)
    return x

def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,nargs=argparse.REMAINDER)
    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if args.config_file != "": cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir): mkdir(output_dir)
    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)
    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    if cfg.MODEL.DEVICE == "cuda": os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True
    def extract_feat_pn(loader):
        ptr = 0
        gall_feat_pos = np.zeros((len(loader.dataset), feature_dim))
        gall_feat_neg = np.zeros((len(loader.dataset), feature_dim))
        for batch_idx, (input, label) in enumerate(loader):
            batch_num = input.size(0)
            model.eval()
            with torch.no_grad():
                input = input.to('cuda')
                feat_pos, feat_neg = model(input)
                feat_pos = norm(feat_pos)
                feat_neg = norm(feat_neg)
                gall_feat_pos[ptr:ptr + batch_num, :] = feat_pos.detach().cpu().numpy()
                gall_feat_neg[ptr:ptr + batch_num, :] = feat_neg.detach().cpu().numpy()
                ptr = ptr + batch_num
        return gall_feat_pos, gall_feat_neg





    feature_dim = 2048
    def extract_feat(loader):
        ptr = 0
        gall_feat = np.zeros((len(loader.dataset), feature_dim))
        for batch_idx, (input, label) in enumerate(loader):
            batch_num = input.size(0)
            model.eval()
            with torch.no_grad():
                input = input.to('cuda')

                # feat_pos = model(input)
                feat_pos, feat_neg = model(input)

                # feat = torch.cat((feat_pos, feat_neg), dim=1)
                # feat = (feat_neg + feat_pos)/2
                # feat = torch.cat((norm(feat_pos), norm(feat_neg)), dim=1)

                feat = norm(feat_pos)

                gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                ptr = ptr + batch_num
        return gall_feat



    '==============================================================================================='
    model = build_model(cfg, 395)
    model.load_state_dict(torch.load(cfg.TEST.WEIGHT))
    model.to('cuda')
    '==============================================================================================='
    data_path = '/data1/lidg/reid_dataset/IV-ReID/SYSU'
    query_img, query_label, query_cam = process_query_sysu(data_path, mode='all')
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode='all', trial=0)


    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    hight, width = 256, 128  #   (384, 256)   (256, 128)  (224,224)
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((hight, width)),
        transforms.ToTensor(),
        normalize,])

    '==============================================================================================='
    queryset = TestData(query_img, query_label, transform=transform_test)
    query_loader = data.DataLoader(queryset, batch_size=29, shuffle=False, num_workers=4)
    print('Extracting Query Feature...')
    query_feat = extract_feat(query_loader)
    '==============================================================================================='

    all_cmc = 0
    all_mAP = 0
    acc = np.zeros((10, 4))
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode='all', trial=trial)

        '==============================================================================================='
        trial_gallset = TestData(gall_img, gall_label, transform=transform_test)
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=29, shuffle=False, num_workers=4)
        print('Extracting Gallery Feature...')
        gall_feat = extract_feat(trial_gall_loader)
        '==============================================================================================='


        distmat = np.matmul(query_feat, np.transpose(gall_feat))



        cmc, mAP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)







        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
        print('Test Trial: {}'.format(trial))
        print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(cmc[0], cmc[4], cmc[9], cmc[19]))
        print('mAP: {:.2%}'.format(mAP))
        acc[trial][0] = float('%.4f' % cmc[0])
        acc[trial][1] = float('%.4f' % cmc[9])
        acc[trial][2] = float('%.4f' % cmc[19])
        acc[trial][3] = float('%.4f' % mAP)
    cmc = all_cmc / 10
    mAP = all_mAP / 10
    print('All Average:')
    print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(cmc[0], cmc[4], cmc[9], cmc[19]))
    print('mAP: {:.2%}'.format(mAP))
    print(np.mean(acc, 0))
    print(np.std(acc, 0))

if __name__ == '__main__':
    main()
