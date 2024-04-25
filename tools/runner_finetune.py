import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

import numpy as np
from datasets import data_transforms
from extensions.pointnet2 import pointnet2_utils as pt_utils
from extensions.pointops.functions import pointops
from torchvision import transforms
from utils.grad_utils import IterativePercentile

import torch.nn.functional as F


train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


class Seg_Metric:
    def __init__(self, macc, miou=0., mdsc=0.):
        self.macc = macc
        self.miou = miou
        self.mdsc = mdsc
    
    def better_than(self, other):
        this_score = self.macc + self.miou + self.mdsc
        other_score = other.macc + other.miou + other.mdsc
        if this_score > other_score:
            return True
        else:
            return False
    
    def state_dict(self):
        _dict = dict()
        _dict['macc'] = self.macc
        _dict['mdsc'] = self.mdsc
        _dict['miou'] = self.miou
        return _dict


def run_net(args, config, train_writer=None, val_writer=None):
    # get logger
    logger = get_logger(args.log_name)

    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # build model
    model = builder.model_builder(config.model)
    
    # parameter setting
    start_epoch = 0
    best_metrics = Seg_Metric(0.)
    metrics = Seg_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(model, args, logger = logger)
        best_metrics = Seg_Metric(best_metrics)
    else:
        if args.ckpts is not None:
            model.load_model_from_ckpt(args.ckpts)
            # model.load_model_from_ckpt_direct(args.ckpts)
        else:
            print_log('Training from scratch', logger = logger)

    if args.use_gpu:    
        model.to(args.local_rank)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        pass
        # print_log('Using Data parallel ...' , logger = logger)
        # model = nn.DataParallel(model).cuda()

    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    # training
    model.zero_grad()

    gradclip_percentile = config.get('gradclip_percentile', -1)
    if gradclip_percentile > 0:
        grad_history = IterativePercentile(p=gradclip_percentile)
    else:
        grad_history = None

    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        n_batches = len(train_dataloader)

        grad_clip_val = config.grad_norm_clip
        
        for idx, (points, cls, labels, class_weights) in enumerate(train_dataloader):

            # parameter update
            num_iter += 1
            n_itr = epoch * n_batches + idx
            data_time.update(time.time() - batch_start_time)

            # data to gpu and augmentation
            points, cls, labels, class_weights = points.cuda(), cls.cuda(), labels.cuda(), class_weights.cuda()
            points = train_transforms(points)

            # forward
            logit = model(points, to_categorical(cls, 2))

            # get loss
            loss, acc = model.get_loss_acc(logit, labels, class_weights)

            _loss = loss.mean()
            _loss.backward()

            loss = _loss

            # backward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    if grad_history is not None:
                        grad_norm = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None]).norm().item()
                        grad_clip_val = grad_history.add(grad_norm)
                        if train_writer is not None:
                            train_writer.add_scalar('Loss/Batch/clip_val', grad_clip_val, n_itr)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val, norm_type=2)
                    if train_writer is not None:
                        train_writer.add_scalar('Loss/grad_norm', grad_norm.item(), n_itr)
                num_iter = 0
                optimizer.step()
                model.zero_grad()
            
            # record statistics
            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
                losses.update([loss.item(), acc.item()])
            else:
                losses.update([loss.item(), acc.item()])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss+Acc = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Loss_2', losses.avg(1), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)
        
        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics = validate(model, test_dataloader, epoch, val_writer, args, config, logger=logger)

            better = metrics.better_than(best_metrics)
            # Save ckeckpoints
            if better:
                best_metrics = metrics
                builder.save_checkpoint(model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)

        builder.save_checkpoint(model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
    
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

    return metrics, best_metrics
    

def validate(model, test_dataloader, epoch, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    model.eval()  # set model to eval mode

    mandible_metric, maxillary_metric  = [], []

    with torch.no_grad():
        for idx, (points, cls, labels, points_whole, labels_whole, center, scale, _) in enumerate(test_dataloader):
            points, cls, labels = points.cuda(), cls.cuda(), labels.cuda()
            points_whole, labels_whole, center, scale = points_whole.cuda(), labels_whole.cuda(), center.cuda(), scale.cuda()
            logits = model(points, to_categorical(cls, 2))

            pred_whole = get_pred_whole(logits, points, points_whole, center, scale)
            macc, miou, mdsc = get_seg_metrics(pred_whole.detach(), labels_whole)

            if cls.squeeze(0) == 0:
                mandible_metric.append(torch.tensor([macc, miou, mdsc]))
            else:
                maxillary_metric.append(torch.tensor([macc, miou, mdsc]))
        
        mandible_metric = torch.stack(mandible_metric, dim=0).cpu().numpy()
        maxillary_metric = torch.stack(maxillary_metric, dim=0).cpu().numpy()
        whole_metric = np.concatenate((mandible_metric, maxillary_metric), axis=0)
        
        mandible_macc, mandible_miou, mandible_mdsc = mandible_metric.mean(axis=0)[0],\
            mandible_metric.mean(axis=0)[1], mandible_metric.mean(axis=0)[2]
        maxillary_macc, maxillary_miou, maxillary_mdsc = maxillary_metric.mean(axis=0)[0], \
            maxillary_metric.mean(axis=0)[1], maxillary_metric.mean(axis=0)[2]
        whole_macc, whole_miou, whole_mdsc = whole_metric.mean(axis=0)[0], \
            whole_metric.mean(axis=0)[1], whole_metric.mean(axis=0)[2]
        
        print_log('[Validation] EPOCH: %d  macc = %.4f  miou = %.4f  mdsc = %.4f' % \
                  (epoch, whole_macc, whole_miou, whole_mdsc), logger=logger)
        print_log('[Validation] EPOCH: %d  mandi_macc = %.4f  mandi_miou = %.4f  mandi_mdsc = %.4f' % \
                  (epoch, mandible_macc, mandible_miou, mandible_mdsc), logger=logger)
        print_log('[Validation] EPOCH: %d  maxil_macc = %.4f  maxil_miou = %.4f  maxil_mdsc = %.4f' % \
                  (epoch, maxillary_macc, maxillary_miou, maxillary_mdsc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/macc', whole_macc, epoch)
        val_writer.add_scalar('Metric/miou', whole_miou, epoch)
        val_writer.add_scalar('Metric/mdsc', whole_mdsc, epoch)

    return Seg_Metric(whole_macc, whole_miou, whole_mdsc)


def get_pred_whole(logits, points, points_whole, center, scale):
    logits = F.softmax(logits, dim=1)
    points = points * scale + center
    dist, idx = pt_utils.three_nn(points_whole, points)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm
    logits_whole = pt_utils.three_interpolate(logits, idx, weight)
    pred_whole = logits_whole.argmax(dim=1)
    
    return pred_whole


def get_seg_metrics(pred_whole, labels_whole):
    pred_whole = pred_whole.squeeze().cpu()
    labels_whole = labels_whole.squeeze().cpu()

    unq_labels = torch.unique(labels_whole).cpu().numpy()
    acc, iou, dsc = [], [], []

    for jcls in unq_labels:

        jcls_and = torch.logical_and(pred_whole==jcls, labels_whole==jcls).sum()
        jcls_or = torch.logical_or(pred_whole==jcls, labels_whole==jcls).sum()

        iou.append((jcls_and / jcls_or).float())
        dsc.append((2*iou[-1] / (1 + iou[-1])))
    
    acc = (pred_whole == labels_whole).sum() / (labels_whole.view(-1).shape[0])
    miou = np.array(iou).mean()
    mdsc = np.array(dsc).mean()

    return acc, miou, mdsc


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(model, args.ckpts, logger = logger) # for finetuned transformer
    # model.load_model_from_ckpt(args.ckpts) # for BERT
    if args.use_gpu:
        model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()
     
    test(model, test_dataloader, args, config, logger=logger)


def test(model, test_dataloader, args, config, logger = None):

    model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            logits = model(points)
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[TEST] acc = %.4f' % acc, logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

        if args.no_vote:
            return

        print_log(f"[TEST_VOTE]", logger = logger)
        acc = 0.
        for time in range(1, 10):
            this_acc = test_vote(model, test_dataloader, 1, None, args, config, logger=logger, times=time)
            if acc < this_acc:
                acc = this_acc

        print_log('[TEST_VOTE] acc = %.4f' % acc, logger=logger)


def test_net_extract_feat(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    model = builder.model_builder(config.model)
    # load checkpoints
    # builder.load_model(model, args.ckpts, logger = logger) # for finetuned transformer

    # load state dict
    if args.ckpts is not None:
        state_dict = torch.load(args.ckpts, map_location='cpu')
        # parameter resume of base model
        if state_dict.get('model') is not None:
            base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
        elif state_dict.get('model') is not None:
            base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
        else:
            raise RuntimeError('mismatch of ckpt weight')

        try:
            model.load_state_dict(base_ckpt, strict = True)
        except:
            print('fail to directly load, attempting to load as mae...')
            model.load_model_from_ckpt(args.ckpts) # for BERT
    if args.use_gpu:
        model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints

    features = []
    labels = []
    num_features = 0

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            feats_batch = model(points, return_feature=True)
            features.append(feats_batch.detach())
            labels.append(label.detach())
            num_features += feats_batch.shape[0]

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    features_np = features.data.cpu().numpy()
    labels_np = labels.data.cpu().numpy()
    np.savez(f'{args.experiment_path}/features', features=features_np, labels=labels_np)
