# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from functools import partial
import logging
import numpy as np
import os
import random

import torch
import torch.distributed as dist

from dinov3.data import DatasetWithEnumeratedTargets, SamplerType, make_data_loader
import dinov3.distributed as distributed
from dinov3.eval.segmentation.datasets import build_segmentation_dataset
from dinov3.eval.segmentation.eval import evaluate_segmentation_model
from dinov3.eval.segmentation.loss import MultiSegmentationLoss
from dinov3.eval.segmentation.metrics import SEGMENTATION_METRICS
from dinov3.eval.segmentation.models import build_segmentation_decoder
from dinov3.eval.segmentation.schedulers import build_scheduler
from dinov3.eval.segmentation.transforms import make_segmentation_eval_transforms, make_segmentation_train_transforms
from dinov3.logging import MetricLogger, SmoothedValue

logger = logging.getLogger("dinov3")


class InfiniteDataloader:
    def __init__(self, dataloader: torch.utils.data.DataLoader):
        self.dataloader = dataloader
        self.data_iterator = iter(dataloader)
        self.sampler = dataloader.sampler
        if not hasattr(self.sampler, "epoch"):
            self.sampler.epoch = 0  # type: ignore

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return len(self.dataloader)

    def __next__(self):
        try:
            data = next(self.data_iterator)
        except StopIteration:
            self.sampler.epoch += 1
            self.data_iterator = iter(self.dataloader)
            data = next(self.data_iterator)
        return data


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.
    The seed of each worker equals to num_worker * rank + worker_id + user_seed
    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def validate(
    segmentation_model: torch.nn.Module,
    val_dataloader,
    device,
    autocast_dtype,
    eval_res,
    eval_stride,
    decoder_head_type,
    num_classes,
    reduce_zero_label,
    global_step,
    metric_to_save,
    current_best_metric_to_save_value,
):
    new_metric_values_dict = evaluate_segmentation_model(
        segmentation_model,
        val_dataloader,
        device,
        eval_res,
        eval_stride,
        decoder_head_type,
        num_classes,
        autocast_dtype,
        reduce_zero_label,
    )
    logger.info(f"Step {global_step}: {new_metric_values_dict}")
    segmentation_model.module.set_trainable_mode()
    is_better = False
    if new_metric_values_dict[metric_to_save] > current_best_metric_to_save_value:
        is_better = True
    return is_better, new_metric_values_dict


def train_step(
    segmentation_model: torch.nn.Module,
    batch,
    device,
    scaler,
    optimizer,
    optimizer_gradient_clip,
    scheduler,
    criterion,
    model_dtype,
    global_step,
):
    # a) load batch
    batch_img, (_, gt) = batch
    batch_img = batch_img.to(device)  # B x C x h x w 把图片和标签搬到 GPU
    gt = gt.to(device)  # B x (num_classes if multilabel) x h x w 把图片和标签搬到 GPU
    optimizer.zero_grad(set_to_none=True)#再把优化器梯度清掉：

    # b) forward pass
    with torch.autocast("cuda", dtype=model_dtype, enabled=True if model_dtype is not None else False):
        pred = segmentation_model(batch_img)  # B x num_classes x h x w
        gt = gt.long()
        if gt.ndim == 4 and gt.shape[1] == 1:
            gt = gt[:, 0]

    # c) compute loss 我之前还担心损失精度 
    if gt.shape[-2:] != pred.shape[-2:]:
        pred = torch.nn.functional.interpolate(input=pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
    loss = criterion(pred, gt)

    # d) optimization
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(segmentation_model.module.parameters(), optimizer_gradient_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(segmentation_model.module.parameters(), optimizer_gradient_clip)
        optimizer.step()

    if global_step > 0:  # inheritance from old mmcv code
        scheduler.step()

    return loss


def train_segmentation(
    backbone,
    config,
):
    assert config.decoder_head.type == "linear", "Only linear head is supported for training"
    # 1- load the segmentation decoder 判断头的类型是否正确
    logger.info("Initializing the segmentation model")
    segmentation_model = build_segmentation_decoder(
        backbone,
        config.decoder_head.backbone_out_layers,
        config.decoder_head.type,
        num_classes=config.decoder_head.num_classes,
        hidden_dim=config.decoder_head.hidden_dim,
        autocast_dtype=config.model_dtype.autocast_dtype,
        dropout=config.decoder_head.dropout,
        input_adapter_in_channels=config.input_adapter.in_channels,
        input_adapter_mode=config.input_adapter.mode,
    )
    #分布式分配GPU资源
    global_device = distributed.get_rank()
    local_device = torch.cuda.current_device()
    segmentation_model = torch.nn.parallel.DistributedDataParallel(
        segmentation_model.to(local_device), device_ids=[local_device]
    )  # should be local rank
    #保存可训练参数的迭代器，后续统计数量
    model_parameters = filter(lambda p: p.requires_grad, segmentation_model.parameters())
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in model_parameters)}")

    # 2- create data transforms + dataloaders
    train_transforms = make_segmentation_train_transforms(
        img_size=config.transforms.train.img_size,
        #随机增强缩放范围，最终会被裁剪到img_size大小
        random_img_size_ratio_range=config.transforms.train.random_img_size_ratio_range,
        crop_size=config.transforms.train.crop_size,
        flip_prob=config.transforms.train.flip_prob,
        reduce_zero_label=config.eval.reduce_zero_label,
        enable_photometric_distortion=config.transforms.train.enable_photometric_distortion,
        label_values=config.datasets.label_values,
        mean=config.transforms.mean,
        std=config.transforms.std,
    )
    val_transforms = make_segmentation_eval_transforms(
        #固定缩放尺寸
        img_size=config.transforms.eval.img_size,
        #推理模式（滑窗或者全图）验证时不做随机增强，只做固定预处理，保证评估结果准确
        inference_mode=config.eval.mode,
        label_values=config.datasets.label_values,
        #归一化规则
        mean=config.transforms.mean,
        std=config.transforms.std,
    )

    train_dataset = DatasetWithEnumeratedTargets(build_segmentation_dataset(config, split="train", transforms=train_transforms))
    #分布式采样器，保证每个GPU看到不同的数据
    train_sampler_type = None
    if distributed.is_enabled():
        train_sampler_type = SamplerType.DISTRIBUTED
    # worker_init_fn需要传入num_workers, rank, seed等参数，因此使用partial函数固定部分参数
    init_fn = partial(
        worker_init_fn, num_workers=config.num_workers, rank=global_device, seed=config.seed + global_device
    )
    #数据加载器，使用InfiniteDataloader包装，使其成为一个无限迭代器，避免每个epoch结束时需要重新创建迭代器
    train_dataloader = InfiniteDataloader(
        make_data_loader(
            dataset=train_dataset,
            batch_size=config.bs,
            num_workers=config.num_workers,
            sampler_type=train_sampler_type,
            shuffle=True,
            persistent_workers=False,
            worker_init_fn=init_fn,
        )
    )

    val_dataset = DatasetWithEnumeratedTargets(build_segmentation_dataset(config, split="val", transforms=val_transforms))
    #同上，但验证集不需要shuffle
    val_sampler_type = None
    #验证集的分布式采样器，保证每个GPU看到不同的数据
    if distributed.is_enabled():
        val_sampler_type = SamplerType.DISTRIBUTED
    #
    val_dataloader = make_data_loader(
        dataset=val_dataset,
        batch_size=1,
        num_workers=config.num_workers,
        sampler_type=val_sampler_type,
        drop_last=False,
        shuffle=False,
        persistent_workers=True,#保持数据加载进程常驻
    )

    # 3- define and create scaler, optimizer, scheduler, loss
    #自动混合精度，只有当模型数据类型配置了自动混合精度的dtype时才启用
    scaler = None
    if config.model_dtype.autocast_dtype is not None:
        scaler = torch.amp.GradScaler("cuda")

#优化器（AdamW）初始化
    optimizer = torch.optim.AdamW(
        [
            {
                "params": filter(lambda p: p.requires_grad, segmentation_model.parameters()),
                "lr": config.optimizer.lr,
                "betas": (config.optimizer.beta1, config.optimizer.beta2),
                "weight_decay": config.optimizer.weight_decay,
            }
        ]
    )
#  学习率调度器初始化，支持多种调度策略（如CosineAnnealingLR、StepLR等），根据配置选择
    scheduler = build_scheduler(
        config.scheduler.type,
        optimizer=optimizer,
        lr=config.optimizer.lr,
        total_iter=config.scheduler.total_iter,
        constructor_kwargs=config.scheduler.constructor_kwargs,
    )
#损失函数，结合Dice Loss和Cross Entropy Loss，权重可调，根据
    criterion = MultiSegmentationLoss(
        diceloss_weight=config.train.diceloss_weight,
        celoss_weight=config.train.celoss_weight,
        class_weights=config.train.class_weights,
    )
# 记录训练总步数和当前最佳指标值的字典，后续用于评估和保存模型
    total_iter = config.scheduler.total_iter
    global_step = 0
    global_best_metric_values = {metric: 0.0 for metric in SEGMENTATION_METRICS}

#训练日志初始化，使用MetricLogger记录损失和指标，设置日志输出频率和格式
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", SmoothedValue(window_size=4, fmt="{value:.3f}"))
#训练主循环，使用MetricLogger记录每个batch的损失，并每隔一定步数进行一次验证，评估模型性能并根据指定的指标判断是否保存最佳模型
    for batch in metric_logger.log_every(
        train_dataloader,
        50,
        header="Train: ",
        start_iteration=global_step,
        n_iterations=total_iter,
    ):
        if global_step >= total_iter:
            break
        loss = train_step(
            segmentation_model,
            batch,
            local_device,
            scaler,
            optimizer,
            config.optimizer.gradient_clip,
            scheduler,
            criterion,
            config.model_dtype.autocast_dtype,
            global_step,
        )
        global_step += 1
        metric_logger.update(loss=loss)
#每隔eval_interval步进行一次验证，评估模型性能并根据指定的指标判断是否保存最佳模型
        if global_step % config.eval.eval_interval == 0:
            dist.barrier()
            is_better, best_metric_values_dict = validate(
                segmentation_model,
                val_dataloader,
                local_device,
                config.model_dtype.autocast_dtype,
                config.eval.crop_size,
                config.eval.stride,
                config.decoder_head.type,
                config.decoder_head.num_classes,
                config.eval.reduce_zero_label,
                global_step,
                config.metric_to_save,
                global_best_metric_values[config.metric_to_save],
            )
            if is_better:
                logger.info(f"New best metrics at Step {global_step}: {best_metric_values_dict}")
                global_best_metric_values = best_metric_values_dict

    # one last validation only if the number of total iterations is NOT divisible by eval interval
    if total_iter % config.eval.eval_interval:
        dist.barrier()
        is_better, best_metric_values_dict = validate(
            segmentation_model,
            val_dataloader,
            local_device,
            config.model_dtype.autocast_dtype,
            config.eval.crop_size,
            config.eval.stride,
            config.decoder_head.type,
            config.decoder_head.num_classes,
            config.eval.reduce_zero_label,
            global_step,
            config.metric_to_save,
            global_best_metric_values[config.metric_to_save],
        )
        if is_better:
            logger.info(f"New best metrics at Step {global_step}: {best_metric_values_dict}")
            global_best_metric_values = best_metric_values_dict
    logger.info("Training is done!")
    # Save only trainable modules (decoder and optional input adapter).
    torch.save(
        {
            "model": segmentation_model.module.get_finetune_state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        os.path.join(config.output_dir, "model_final.pth"),
    )
    logger.info(f"Final best metrics: {global_best_metric_values}")
    return global_best_metric_values
