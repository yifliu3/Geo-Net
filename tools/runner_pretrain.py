import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from sklearn.svm import LinearSVC
import numpy as np
from torchvision import transforms
from datasets import data_transforms
from utils.grad_utils import IterativePercentile

train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    
    # build dataset
    train_sampler, train_dataloader = builder.dataset_builder(args, config.dataset.train)

    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        pass
    
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    gradclip_percentile = config.get('gradclip_percentile', -1)
    if gradclip_percentile > 0:
        grad_history = IterativePercentile(p=gradclip_percentile)
    else:
        grad_history = None 

    # trainval
    # training
    base_model.zero_grad()
    fixed_sample = None
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss1', 'Loss2', 'Loss3'])

        num_iter = 0
        grad_clip_val = config.grad_norm_clip

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, data in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            elif dataset_name == 'ModelNet':
                points = data[0].cuda()
                points = misc.fps(points, npoints)
            elif dataset_name.startswith('ScanNet'):
                points = data.cuda()
            elif dataset_name.startswith('TeethSeg'):
                points = data[0].cuda()
                curvatures = data[1].cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')
            
            assert points.size(1) == npoints
            points = train_transforms(points)

            if args.overfit_single_batch:
                if fixed_sample is None:
                    fixed_sample = points.clone()
                else:
                    points = fixed_sample.clone()

            loss_list = base_model(points, curvatures)

            _loss = sum(loss_list)
            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    if grad_history is not None:
                        grad_norm = torch.cat([p.grad.view(-1) for p in base_model.parameters() if p.grad is not None]).norm().item()
                        grad_clip_val = grad_history.add(grad_norm)
                        if train_writer is not None:
                            train_writer.add_scalar('Loss/Batch/clip_val', grad_clip_val, n_itr)
                    grad_norm = torch.nn.utils.clip_grad_norm_(base_model.parameters(), grad_clip_val, norm_type=2)
                    if train_writer is not None:
                        train_writer.add_scalar('Loss/grad_norm', grad_norm.item(), n_itr)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss_list = [dist_utils.reduce_tensor(loss, args) for loss in loss_list]
                losses.update([loss.item() for loss in loss_list])
            else:
                losses.update([loss.item() for loss in loss_list])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                for i in range(len(loss_list)):
                    train_writer.add_scalar('Loss/Batch/Loss_{}'.format(i), loss_list[i].item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
            
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            for i in range(len(loss_list)):
                train_writer.add_scalar('Loss/Epoch/Loss_{}'.format(i), losses.avg(i), epoch)
           
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
        
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def test_net():
    pass