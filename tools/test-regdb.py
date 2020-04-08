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

import logging

import torch.nn as nn
from ignite.engine import Engine

from utils.reid_metric import R1_mAP, R1_mAP_reranking


from utils.data_loader import SYSUData, RegDBData, TestData
from utils.data_manager import *
from utils.eval_metrics import eval_sysu, eval_regdb
from utils.model import embed_net
from utils.utils import *

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



def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    #import pdb;pdb.set_trace()

    model = build_model(cfg, 412)
    model.load_state_dict(torch.load(cfg.TEST.WEIGHT))
    model.to('cuda')

    '==============================================================================================='

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128)),   #  384, 256
        transforms.ToTensor(),
        #normalize,
    ])


    data_path = '/data1/lidg/reid_dataset/IV-ReID/RegDB/'

    query_img, query_label = process_test_regdb(data_path, trial=0, modal='thermal')   #  thermal    visible
    gall_img, gall_label = process_test_regdb(data_path, trial=0, modal='visible')

    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(384, 256))
    gall_loader = data.DataLoader(gallset, batch_size=1, shuffle=False, num_workers=4)


    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(384, 256))
    query_loader = data.DataLoader(queryset, batch_size=1, shuffle=False, num_workers=4)

    feature_dim = 2048

    def extract_gall_feat(gall_loader):
        print('Extracting Gallery Feature...')
        ptr = 0
        gall_feat = np.zeros((ngall, feature_dim))

        rgbs = np.zeros((ngall,256,128,3))
        learns = np.zeros((ngall,256,128,3))

        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)

            model.eval()
            with torch.no_grad():
                input = input.to('cuda')

                #feat = model(input)
                feat, rgb, x = model(input)

                rgbs[ptr:ptr+batch_num,:] = rgb
                learns[ptr:ptr+batch_num,:] = x

                gall_feat[ptr:ptr+batch_num,:] = feat.detach().cpu().numpy()
                ptr = ptr + batch_num


        from torchvision.transforms import ToPILImage
        import matplotlib.pyplot as plt
        img1 = np.mean(rgbs,axis=0)
        img2 = np.mean(learns,axis=0)

        #import pdb;pdb.set_trace()

        img1 = ToPILImage()(img1.astype(np.uint8))
        img2 = ToPILImage()(img2.astype(np.uint8))

        img1.save('regdb_RGB.jpg')
        img2.save('regdb_X.jpg')
        import pdb; pdb.set_trace()


        return gall_feat

    def extract_query_feat(query_loader):
        print('Extracting Query Feature...')
        ptr = 0
        query_feat = np.zeros((nquery, feature_dim))
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            model.eval()
            with torch.no_grad():
                input = input.to('cuda')
                feat = model(input)
                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                ptr = ptr + batch_num
        return query_feat


    gall_feat = extract_gall_feat(gall_loader)

    query_feat = extract_query_feat(query_loader)

# fc feature
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    cmc, mAP = eval_regdb(-distmat, query_label, gall_label)


    print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(cmc[0], cmc[4], cmc[9], cmc[19]))
    print('mAP: {:.2%}'.format(mAP))

if __name__ == '__main__':
    main()
