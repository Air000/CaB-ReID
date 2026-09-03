import torch
import torch.nn as nn
import torch.nn.functional as F


class PartMaskProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def _binary(self, masks):
        if self.config.mask_threshold > 0:
            return (masks >= self.config.mask_threshold).to(masks.dtype)
        return (masks > 0).to(masks.dtype)

    def forward(self, masks, view_labels=None):
        if masks is None:
            return None, None
        masks = masks.float()
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        if masks.shape[1] < 2:
            padding = masks.new_zeros(masks.shape[0], 2 - masks.shape[1], *masks.shape[2:])
            masks = torch.cat((masks, padding), dim=1)
        masks = masks[:, :2].clamp(0.0, 1.0)
        binary = self._binary(masks)
        cab_area = binary[:, 0].flatten(1).mean(1)
        body_area = binary[:, 1].flatten(1).mean(1)

        if self.config.body_exclusive_from_cab:
            masks = masks.clone()
            masks[:, 1:2] = binary[:, 1:2] * (1.0 - binary[:, 0:1])
        if self.training and self.config.mask_dropout > 0:
            keep = torch.rand(masks.shape[0], 2, 1, 1, device=masks.device)
            masks = masks * (keep >= self.config.mask_dropout).to(masks.dtype)

        if self.config.mask_threshold > 0:
            thresholded = self._binary(masks)
            visibility = thresholded
            if self.config.binary_pooling:
                masks = thresholded
        else:
            visibility = masks
        area = visibility.flatten(2).mean(2)
        valid = area >= self.config.min_part_ratio if self.config.part_validity_enabled else torch.ones_like(area, dtype=torch.bool)
        if self.config.part_validity_enabled and self.config.body_valid_area_minus_cab:
            valid[:, 1] = (body_area - cab_area).clamp_min(0.0) >= self.config.min_part_ratio
        if self.config.part_validity_enabled and self.config.cab_invalid_viewids and view_labels is not None:
            view_labels = view_labels.to(device=valid.device).view(-1)
            if view_labels.numel() != valid.shape[0]:
                raise RuntimeError("Expected one view label per sample")
            invalid_ids = torch.tensor(self.config.cab_invalid_viewids, device=valid.device, dtype=view_labels.dtype)
            valid[:, 0] &= ~(view_labels[:, None] == invalid_ids[None, :]).any(1)
        return masks * valid.to(masks.dtype).view(-1, 2, 1, 1), valid


class MaskedTokenPool(nn.Module):
    def __init__(self, pool_beta=0.0):
        super().__init__()
        self.pool_beta = max(0.0, min(1.0, float(pool_beta)))

    @staticmethod
    def patch_mask(mask, token_hw, dtype, device):
        if tuple(mask.shape[-2:]) != tuple(token_hw):
            mask = F.interpolate(mask.float(), size=token_hw, mode="area")
        return mask.flatten(1).to(dtype=dtype, device=device).clamp(0, 1)

    def forward(self, token_features, patch_mask):
        cls_token = token_features[:, 0]
        tokens = token_features[:, 1:]
        weights = patch_mask.to(dtype=tokens.dtype, device=tokens.device)
        denominator = weights.sum(1, keepdim=True)
        pooled = (tokens * weights.unsqueeze(-1)).sum(1) / denominator.clamp_min(1e-6)
        has_region = (denominator > 1e-6).to(dtype=tokens.dtype)
        pooled = pooled * has_region + cls_token * (1.0 - has_region)
        if self.pool_beta:
            pooled = pooled * (1.0 - self.pool_beta) + cls_token * self.pool_beta
        return pooled


class PartFeatureFusion(nn.Module):
    def __init__(self, global_weight, cab_weight, body_weight):
        super().__init__()
        self.global_weight = float(global_weight)
        self.cab_weight = float(cab_weight)
        self.body_weight = float(body_weight)

    def forward(self, global_feature, cab_feature, body_feature, valid):
        denominator = global_feature.new_full((global_feature.shape[0], 1), self.global_weight)
        fused = global_feature * denominator
        if self.cab_weight > 0:
            cab_weight = valid[:, 0].to(global_feature.dtype).view(-1, 1) * self.cab_weight
            fused = fused + cab_feature * cab_weight
            denominator = denominator + cab_weight
        if self.body_weight > 0:
            body_weight = valid[:, 1].to(global_feature.dtype).view(-1, 1) * self.body_weight
            fused = fused + body_feature * body_weight
            denominator = denominator + body_weight
        return fused / denominator.clamp_min(1e-6)
