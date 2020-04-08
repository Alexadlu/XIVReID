# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
import torch.nn.functional as F

# from .triplet_loss import CrossEntropyLabelSmooth_neg
from .triplet_loss_anti import *

from utils.reid_metric import R1_mAP
import numpy as np

global ITER
ITER = 0

softmin = CrossEntropyLabelSmooth_neg(395)
antitriplet = TripletLoss(-0.3)

def normalize(x):
    x = 1. * x / (torch.norm(x, 2, -1, keepdim=True).expand_as(x) + 1e-12)
    return x

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1).expand(m, n)
    yy = torch.pow(y, 2).sum(1).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def triplet(feat1, feat2):

    '=================================================================================='
    distmap = euclidean_dist(feat1, feat2)
    # distmap = -torch.mm(feat1, feat2.permute(1,0).contiguous())
    # distmap = -torch.mm(normalize(feat1), normalize(feat2).permute(1,0).contiguous())
    '=================================================================================='

    distap = torch.diag(distmap)
    distan = distmap + (torch.eye(distmap.shape[0]) * 100).to('cuda')
    distan1 = torch.min(distan, 1)[0]
    distan2 = torch.min(distan, 0)[0]
    distan = torch.min(distan1, distan2)
    cross_triplet_loss = torch.mean(torch.clamp(0.5 + distap - distan, min=0.0))
    return cross_triplet_loss


def create_supervised_trainer(model, optimizer, loss_fn, device=None):
    if device:
        if torch.cuda.device_count() > 1:model = nn.DataParallel(model)
        model.to(device)


    def _update(engine, batch):
        indx = np.arange(batch[0].shape[0]//2)*2
        bs = indx.shape[0]

        model.train()
        optimizer.zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target


        'Baseline========================================================================================='
        target = torch.cat((target, target[indx]), dim=0)
        score, feat = model(img)
        loss_rgb = loss_fn(score[indx], feat[indx], target[indx])
        loss_infrared = loss_fn(score[indx+1], feat[indx+1], target[indx])
        loss_x = loss_fn(score[bs*2:], feat[bs*2:], target[indx])

        visible = feat[indx].unsqueeze(0).unsqueeze(0)
        infrared = feat[indx+1].unsqueeze(0).unsqueeze(0)
        x = feat[bs*2:].unsqueeze(0).unsqueeze(0)
        visible = F.avg_pool2d(visible, kernel_size=(4,1), stride=(4,1)).view(bs//4, -1)
        infrared = F.avg_pool2d(infrared, kernel_size=(4,1), stride=(4,1)).view(bs//4, -1)
        x = F.avg_pool2d(x, kernel_size=(4,1), stride=(4,1)).view(bs//4, -1)

        loss_cmt_iv = triplet(infrared, visible) + triplet(visible, infrared)
        loss_cmt_ix = triplet(infrared, x) + triplet(x, infrared)
        loss = loss_rgb + loss_infrared + loss_x + 0.1*loss_cmt_iv + 0.1*loss_cmt_ix
        'Baseline========================================================================================='




        loss.backward()
        optimizer.step()
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)


def create_supervised_trainer_attention(model, optimizer, loss_fn, device=None):
    if device:
        if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
        model.to(device)


    def triplet_cmt(infrared, visible):
        infrared = infrared.unsqueeze(0).unsqueeze(0)
        visible = visible.unsqueeze(0).unsqueeze(0)
        infrared = F.avg_pool2d(infrared, kernel_size=(4,1), stride=(4,1)).view(infrared.shape[0], -1)
        visible = F.avg_pool2d(visible, kernel_size=(4,1), stride=(4,1)).view(visible.shape[0], -1)
        return triplet(infrared, visible) + triplet(visible, infrared)


    def _update(engine, batch):
        indx = np.arange(batch[0].shape[0]//2)*2
        bs = indx.shape[0]
        BS = bs*2

        model.train()
        optimizer.zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target


        '========================================================================================================================='
        c_p, c_n, f_p, f_n, loss_mask = model(img)

        loss_pos = loss_fn(c_p, f_p, target)
        loss_neg = softmin(c_n, target)

        loss = loss_pos + 0.1*loss_neg + loss_mask

        acc = (c_p.max(1)[1] == target).float().mean()
        loss.backward()
        optimizer.step()
        return loss_neg.item(), acc.item()
        '========================================================================================================================='



    return Engine(_update)


def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn, num_query,start_epoch):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")


    #=============================================================================================#
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    # trainer = create_supervised_trainer_attention(model, optimizer, loss_fn, device=device)
    #=============================================================================================#


    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=5, require_empty=False)
    timer = Timer(average=True)


    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model, 'optimizer': optimizer})
    # trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model.state_dict(),'optimizer': optimizer.state_dict()})


    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)


def create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn, cetner_loss_weight,device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):

        indx = np.arange(batch[0].shape[0] // 2) * 2

        model.train()
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target

        # rgb = img[indx]
        # gray = torch.max(rgb, dim=1, keepdim=True)[0].expand_as(rgb)
        # img = torch.cat((img, gray),dim=0)
        target = torch.cat((target, target[indx]), dim=0)

        score, feat = model(img)

        loss_rgb = loss_fn(score[indx], feat[indx], target[indx])
        loss_infrared = loss_fn(score[indx+1], feat[indx+1], target[indx])
        loss_gray = loss_fn(score[batch[0].shape[0]:], feat[batch[0].shape[0]:], target[indx])

        loss_cross_triplet = 0.1 * trip3(feat)

        #loss = loss_rgb + loss_infrared

        loss = loss_rgb + loss_infrared + loss_gray + loss_cross_triplet

        # print("Total loss is {}, center loss is {}".format(loss, center_criterion(feat, target)))

        loss.backward()
        optimizer.step()
        for param in center_criterion.parameters():
            param.grad.data *= (1. / cetner_loss_weight)
        optimizer_center.step()

        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)


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


def do_train_with_center(cfg,model,center_criterion,train_loader,val_loader,optimizer,optimizer_center,scheduler,loss_fn,num_query,start_epoch):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")

    trainer = create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn, cfg.SOLVER.CENTER_LOSS_WEIGHT, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)

    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer,
                                                                     'optimizer_center': optimizer_center})


    # trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model.state_dict(),
    #                                                                  'optimizer': optimizer.state_dict(),
    #                                                                  'optimizer_center': optimizer_center.state_dict()})

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)