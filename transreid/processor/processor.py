import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from cabreid.memory import ClusterMemoryAMP, compute_cluster_centroids, compute_visible_cluster_centroids, extract_part_image_features

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             part_cluster_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    cab_loss_meter = AverageMeter()
    body_loss_meter = AverageMeter()
    cab_vis_meter = AverageMeter()
    body_vis_meter = AverageMeter()
    acc_meter = AverageMeter()
    part_memory_enabled = bool(cfg.PART_MEMORY.ENABLED)
    cab_loss_weight = float(cfg.PART_MEMORY.CAB_LOSS_WEIGHT)
    body_loss_weight = float(cfg.PART_MEMORY.BODY_LOSS_WEIGHT)

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        cab_loss_meter.reset()
        body_loss_meter.reset()
        cab_vis_meter.reset()
        body_vis_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        cab_memory = None
        body_memory = None
        if part_memory_enabled:
            part_image_features, gt_labels, part_features = extract_part_image_features(model, part_cluster_loader, use_amp=True)
            part_image_features = torch.nn.functional.normalize(part_image_features.float(), dim=1)
            fallback_centroids = compute_cluster_centroids(part_image_features, gt_labels)
            cab_memory = ClusterMemoryAMP(temp=cfg.PART_MEMORY.MEMORY_TEMP, momentum=cfg.PART_MEMORY.MEMORY_MOMENTUM, use_hard=True).to(device)
            body_memory = ClusterMemoryAMP(temp=cfg.PART_MEMORY.MEMORY_TEMP, momentum=cfg.PART_MEMORY.MEMORY_MOMENTUM, use_hard=True).to(device)
            cab_memory.features = compute_visible_cluster_centroids(part_features['cab'], gt_labels, part_features['cab_valid'], fallback_centroids).to(device)
            body_memory.features = compute_visible_cluster_centroids(part_features['body'], gt_labels, part_features['body_valid'], fallback_centroids).to(device)
            logger.info('Create prompt-part memory banks: cab visible = {:.2%}, body visible = {:.2%}'.format(part_features['cab_valid'].float().mean().item(), part_features['body_valid'].float().mean().item()))
        model.train()
        for n_iter, batch in enumerate(train_loader):
            if len(batch) == 5:
                img, mask, vid, target_cam, target_view = batch
                mask = mask.to(device)
            else:
                img, vid, target_cam, target_view = batch
                mask = None
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                if part_memory_enabled:
                    score, feat, part_dict = model(img, target, cam_label=target_cam, view_label=target_view, mask=mask, return_part_features=True)
                else:
                    score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
                    part_dict = None
                loss = loss_fn(score, feat, target, target_cam)
                cab_loss = feat[0].new_tensor(0.0) if isinstance(feat, list) else feat.new_tensor(0.0)
                body_loss = cab_loss.clone()
                cab_vis = cab_loss.clone()
                body_vis = cab_loss.clone()
                if part_memory_enabled and part_dict is not None:
                    cab_idx = part_dict['cab_valid'].bool()
                    body_idx = part_dict['body_valid'].bool()
                    cab_vis = cab_idx.float().mean()
                    body_vis = body_idx.float().mean()
                    if cab_loss_weight > 0 and cab_idx.any():
                        cab_loss = cab_memory(part_dict['cab'][cab_idx], target[cab_idx]) * cab_loss_weight
                        loss = loss + cab_loss
                    if body_loss_weight > 0 and body_idx.any():
                        body_loss = body_memory(part_dict['body'][body_idx], target[body_idx]) * body_loss_weight
                        loss = loss + body_loss

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            cab_loss_meter.update(float(cab_loss.detach().float().cpu()), img.shape[0])
            body_loss_meter.update(float(body_loss.detach().float().cpu()), img.shape[0])
            cab_vis_meter.update(float(cab_vis.detach().float().cpu()), img.shape[0])
            body_vis_meter.update(float(body_vis.detach().float().cpu()), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, CabMem: {:.4f}, BodyMem: {:.4f}, CabVis: {:.4f}, BodyVis: {:.4f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, cab_loss_meter.avg, body_loss_meter.avg,
                                    cab_vis_meter.avg, body_vis_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, batch in enumerate(val_loader):
                        with torch.no_grad():
                            if len(batch) == 7:
                                img, mask, vid, camid, camids, target_view, _ = batch
                                mask = mask.to(device)
                            else:
                                img, vid, camid, camids, target_view, _ = batch
                                mask = None
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view, mask=mask)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, batch in enumerate(val_loader):
                    with torch.no_grad():
                        if len(batch) == 7:
                            img, mask, vid, camid, camids, target_view, _ = batch
                            mask = mask.to(device)
                        else:
                            img, vid, camid, camids, target_view, _ = batch
                            mask = None
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view, mask=mask)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, batch in enumerate(val_loader):
        with torch.no_grad():
            if len(batch) == 7:
                img, mask, pid, camid, camids, target_view, imgpath = batch
                mask = mask.to(device)
            else:
                img, pid, camid, camids, target_view, imgpath = batch
                mask = None
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view, mask=mask)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]

