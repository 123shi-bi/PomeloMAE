import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    num_batches = len(data_loader)

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        group_offset = data_iter_step % accum_iter
        group_size = min(accum_iter, num_batches - (data_iter_step - group_offset))
        update_grad = group_offset + 1 == group_size
        if group_offset == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / num_batches + epoch, args)

        samples = batch[0].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            loss, _, mask, loss_components = model(
                samples,
                mask_ratio=args.mask_ratio,
                use_complementary_mask=args.cmr_complementary_mask,
                mask_strategy=args.cmr_mask_strategy,
                use_cross_view_context=args.cmr_cross_view_context,
                use_token_fusion=args.cmr_token_fusion,
                use_view_consistency=args.cmr_view_consistency,
                view_consistency_weight=args.cmr_view_consistency_weight,
                use_mvtf_cross_layer_fusion=args.mvtf_cross_layer_fusion,
                return_loss_components=True,
            )

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / group_size
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=update_grad)
        if update_grad:
            optimizer.zero_grad()

        if device.type == 'cuda':
            torch.cuda.synchronize(device)

        metric_logger.update(loss=loss_value)
        metric_logger.update(mae_loss=loss_components['mae_loss'].item())
        metric_logger.update(view_consistency_loss=loss_components['view_consistency_loss'].item())
        metric_logger.update(mask_ratio=mask.float().mean().item())

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and update_grad:
            epoch_1000x = int((data_iter_step / num_batches + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('mae_loss', loss_components['mae_loss'].item(), epoch_1000x)
            log_writer.add_scalar(
                'view_consistency_loss', loss_components['view_consistency_loss'].item(), epoch_1000x)
            log_writer.add_scalar('mask_ratio', mask.float().mean().item(), epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
