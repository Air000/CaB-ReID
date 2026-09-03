from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.cuda import amp
import tqdm


class _ClusterUpdate(autograd.Function):
    @staticmethod
    @amp.custom_fwd
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        return inputs.mm(features.t())

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, gradients):
        inputs, targets = ctx.saved_tensors
        input_gradients = gradients.mm(ctx.features) if ctx.needs_input_grad[0] else None
        for feature, target in zip(inputs, targets):
            ctx.features[target] = ctx.momentum * ctx.features[target] + (1.0 - ctx.momentum) * feature
            ctx.features[target] /= ctx.features[target].norm()
        return input_gradients, None, None, None


class _HardClusterUpdate(autograd.Function):
    @staticmethod
    @amp.custom_fwd
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        return inputs.mm(features.t())

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, gradients):
        inputs, targets = ctx.saved_tensors
        input_gradients = gradients.mm(ctx.features) if ctx.needs_input_grad[0] else None
        grouped = defaultdict(list)
        for feature, target in zip(inputs, targets.tolist()):
            grouped[target].append(feature)
        for target, features in grouped.items():
            similarities = [feature.unsqueeze(0).mm(ctx.features[target].unsqueeze(0).t())[0, 0].detach().cpu().numpy() for feature in features]
            hardest = int(np.argmin(np.asarray(similarities)))
            ctx.features[target] = ctx.momentum * ctx.features[target] + (1.0 - ctx.momentum) * features[hardest]
            ctx.features[target] /= ctx.features[target].norm()
        return input_gradients, None, None, None


class ClusterMemory(nn.Module):
    def __init__(self, temperature=0.05, momentum=0.2, hard_update=True):
        super().__init__()
        self.temperature = float(temperature)
        self.momentum = float(momentum)
        self.hard_update = bool(hard_update)
        self.features = None

    def forward(self, inputs, targets):
        inputs = F.normalize(inputs, dim=1)
        momentum = torch.tensor([self.momentum], device=inputs.device)
        update = _HardClusterUpdate if self.hard_update else _ClusterUpdate
        logits = update.apply(inputs, targets, self.features, momentum) / self.temperature
        return F.cross_entropy(logits, targets)


def compute_cluster_centroids(features, labels):
    class_ids = labels.unique()
    class_ids = class_ids[class_ids >= 0]
    centers = torch.zeros((len(class_ids), features.shape[1]), dtype=torch.float32, device=features.device)
    for class_id in class_ids.tolist():
        centers[class_id] = features[labels == class_id].mean(0)
    return F.normalize(centers, dim=1)


def compute_visible_cluster_centroids(features, labels, visible, fallback_centroids=None):
    features = F.normalize(features.float(), dim=1)
    labels = labels.to(features.device)
    visible = visible.bool().to(features.device)
    class_ids = labels.unique()
    class_ids = class_ids[class_ids >= 0]
    centers = torch.zeros((len(class_ids), features.shape[1]), dtype=torch.float32, device=features.device)
    if fallback_centroids is not None:
        fallback_centroids = fallback_centroids.to(features.device).float()
    for class_id in class_ids.tolist():
        indices = torch.where((labels == class_id) & visible)[0]
        if indices.numel():
            centers[class_id] = features[indices].mean(0)
        elif fallback_centroids is not None:
            centers[class_id] = fallback_centroids[class_id]
    return F.normalize(centers, dim=1)


def extract_part_features(model, loader, use_amp=False):
    main_features, cab_features, body_features, labels = [], [], [], []
    cab_valid, body_valid = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Extract CaB-ReID features"):
            if len(batch) == 7:
                image, masks, pid, _camera, _cameras, view, _path = batch
            elif len(batch) == 5:
                image, masks, pid, _camera, view = batch
            else:
                image, pid, _camera, view = batch[:4]
                masks = None
            image = image.cuda()
            masks = masks.cuda() if masks is not None else None
            targets = pid.cuda() if torch.is_tensor(pid) else torch.tensor(pid, device="cuda")
            views = view.cuda() if torch.is_tensor(view) else torch.tensor(view, device="cuda")
            with amp.autocast(enabled=use_amp):
                output = model(
                    image,
                    mask=masks,
                    part_view_label=views,
                    return_part_features=True,
                    feature_mode="memory",
                )
            if isinstance(output, tuple) and len(output) == 5:
                _, _, _, main, parts = output
            elif isinstance(output, tuple) and len(output) == 4:
                _, _, main, parts = output
            elif isinstance(output, tuple) and len(output) == 3:
                _, main, parts = output
                main = main[0] if isinstance(main, list) else main
            elif isinstance(output, tuple) and len(output) == 2:
                main, parts = output
            else:
                raise RuntimeError("Backbone adapter did not return CaB-ReID part features")
            for target, main_feature, cab, body, cab_ok, body_ok in zip(
                targets, main, parts["cab"], parts["body"], parts["cab_valid"], parts["body_valid"]
            ):
                labels.append(target.detach().cpu())
                main_features.append(main_feature.detach().float().cpu())
                cab_features.append(cab.detach().float().cpu())
                body_features.append(body.detach().float().cpu())
                cab_valid.append(cab_ok.detach().cpu().bool())
                body_valid.append(body_ok.detach().cpu().bool())
    return torch.stack(main_features).cuda(), torch.stack(labels).cuda(), {
        "cab": torch.stack(cab_features).cuda(),
        "body": torch.stack(body_features).cuda(),
        "cab_valid": torch.stack(cab_valid).cuda(),
        "body_valid": torch.stack(body_valid).cuda(),
    }


class ClusterMemoryAMP(ClusterMemory):
    def __init__(self, temp=0.05, momentum=0.2, use_hard=True):
        super().__init__(temperature=temp, momentum=momentum, hard_update=use_hard)


extract_part_image_features = extract_part_features
