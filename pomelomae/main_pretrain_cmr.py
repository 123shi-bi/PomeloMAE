import argparse
import datetime
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.datasets_cmr import FourViewPomeloDataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from . import models_mae_cmr
from .engine_pretrain_cmr import train_one_epoch

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.*")


def get_args_parser():
    parser = argparse.ArgumentParser(description='Pre-train PomeloMAE on four views per fruit.')
    parser.add_argument('--batch_size', '--batch-size', default=16, type=int,
                        help='Batch size in pomelo groups. Effective image batch is batch_size * 4 views.')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', '--accum-iter', default=1, type=int)

    parser.add_argument('--model', default='cmr_mae_vit_base_patch16', type=str, metavar='MODEL')
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--mask_ratio', '--mask-ratio', default=0.80, type=float)
    parser.add_argument(
        '--cmr_mask_strategy', default='subject_quadrant_visible',
        choices=('random', 'subject_quadrant_visible'),
        help=('Visible-patch strategy. The default is the final PomeloMAE '
              'cross-view spatial complementary sampler; use random only '
              'for the controlled multi-view component ablation.'))
    parser.add_argument('--cmr_subject_visible_quadrant_ratio', default=0.70, type=float,
                        help='Fraction of visible patches prioritized in the assigned fruit region.')
    parser.add_argument('--norm_pix_loss', action='store_true')
    # The public training entry is intentionally locked to the final method.
    # Compatibility-only constructor fields remain in the model so the retained
    # original checkpoint can still be loaded strictly.
    parser.set_defaults(
        norm_pix_loss=True,
        cmr_complementary_mask=False,
        cmr_subject_patch_threshold=0.25,
        cmr_subject_center_mode='mean',
        cmr_subject_visible_outside_mode='random',
        cmr_cross_view_context=False,
        cmr_center_ratio=0.72,
        cmr_complementary_bias=1.5,
        cmr_context_dropout=0.1,
        cmr_token_fusion=True,
        cmr_view_consistency=False,
        cmr_view_consistency_weight=0.0,
        mvtf_cross_layer_fusion=False,
        mvtf_cross_layer_index=6,
        mvtf_cross_layer_dropout=0.2,
        mvtf_cross_layer_gamma_init=0.3,
        mvtf_cross_layer_fusion_mode='residual')

    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Peak learning rate (default: 1.5e-4).')
    parser.add_argument('--blr', type=float, default=None, metavar='LR',
                        help='If --lr is omitted, scale this base rate by effective images / 256.')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR')
    parser.add_argument('--warmup_epochs', '--warmup-epochs', type=int, default=10, metavar='N')

    parser.add_argument('--data_path', '--data-path', required=True,
                        help='Dataset directory containing the train split.')

    parser.add_argument('--output_dir', '--output-dir', default='./runs/pretrain')
    parser.add_argument('--log_dir', '--log-dir', default=None,
                        help='TensorBoard log directory (defaults to the output directory).')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--num_workers', '--num-workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    return parser


def main(args):
    if args.accum_iter < 1:
        raise ValueError('Gradient accumulation steps must be positive.')
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset_train = FourViewPomeloDataset(
        os.path.join(args.data_path, 'train'),
        transform=transform_train,
        view_suffixes=("a", "b", "c", "d"),
    )
    print(dataset_train)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    log_dir = args.output_dir if args.log_dir is None else args.log_dir
    if global_rank == 0 and log_dir:
        log_writer = SummaryWriter(log_dir=log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if len(data_loader_train) == 0:
        raise ValueError('The training split contains fewer complete four-view groups than one batch.')

    model = models_mae_cmr.__dict__[args.model](
        norm_pix_loss=args.norm_pix_loss,
        center_ratio=args.cmr_center_ratio,
        complementary_bias=args.cmr_complementary_bias,
        context_dropout=args.cmr_context_dropout,
        cross_layer_index=args.mvtf_cross_layer_index,
        cross_layer_dropout=args.mvtf_cross_layer_dropout,
        cross_layer_gamma_init=args.mvtf_cross_layer_gamma_init,
        cross_layer_fusion_mode=args.mvtf_cross_layer_fusion_mode,
        subject_patch_threshold=args.cmr_subject_patch_threshold,
        subject_center_mode=args.cmr_subject_center_mode,
        subject_visible_quadrant_ratio=args.cmr_subject_visible_quadrant_ratio,
        subject_visible_outside_mode=args.cmr_subject_visible_outside_mode,
    )

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * 4 * args.accum_iter * misc.get_world_size()

    if args.lr is None:
        args.lr = 1.5e-4 if args.blr is None else args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective image batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler(enabled=device.type == 'cuda')

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and epoch + 1 == args.epochs:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if log_writer is not None:
        log_writer.close()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
