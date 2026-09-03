import torch
import torch.nn as nn

from .pooling import MaskedTokenPool, PartFeatureFusion, PartMaskProcessor


class CaBReIDModule(nn.Module):
    """Backbone-independent Cab/Body pooling and fusion module."""

    def __init__(self, config, region_masker=None):
        super().__init__()
        self.config = config
        self.region_masker = region_masker
        self.mask_processor = PartMaskProcessor(config)
        self.token_pool = MaskedTokenPool(config.pool_beta)
        self.fusion = PartFeatureFusion(config.global_weight, config.cab_weight, config.body_weight)

    def forward(
        self,
        token_streams,
        global_feature,
        masks=None,
        image=None,
        view_labels=None,
        token_hw=None,
        projector=None,
    ):
        if torch.is_tensor(token_streams):
            token_streams = (token_streams,)
        if self.config.online_mask:
            clean_image = masks[:, 2:5] if masks is not None and masks.dim() == 4 and masks.shape[1] >= 5 else image
            if self.region_masker is None:
                raise RuntimeError("Online masking is enabled but no PromptRegionMasker was provided")
            masks = self.region_masker(clean_image)
        masks, valid = self.mask_processor(masks, view_labels=view_labels)
        if masks is None:
            unavailable = torch.zeros(global_feature.shape[0], dtype=torch.bool, device=global_feature.device)
            return global_feature, {
                "global": global_feature,
                "cab": global_feature,
                "body": global_feature,
                "cab_valid": unavailable,
                "body_valid": unavailable,
            }

        if token_hw is None:
            patch_count = token_streams[0].shape[1] - 1
            token_h = int(patch_count ** 0.5)
            token_hw = (token_h, patch_count // token_h)
        cab_mask = self.token_pool.patch_mask(
            masks[:, 0:1], token_hw, token_streams[0].dtype, token_streams[0].device
        )
        body_mask = self.token_pool.patch_mask(
            masks[:, 1:2], token_hw, token_streams[0].dtype, token_streams[0].device
        )
        cab_parts = tuple(self.token_pool(stream, cab_mask) for stream in token_streams)
        body_parts = tuple(self.token_pool(stream, body_mask) for stream in token_streams)
        if projector is None:
            if len(cab_parts) != 1:
                raise ValueError("Multiple token streams require a part projector")
            cab_feature, body_feature = cab_parts[0], body_parts[0]
        else:
            cab_feature, body_feature = projector(cab_parts), projector(body_parts)
        valid = valid.to(device=global_feature.device)
        fused = self.fusion(global_feature, cab_feature, body_feature, valid)
        return fused, {
            "global": global_feature,
            "cab": cab_feature,
            "body": body_feature,
            "cab_valid": valid[:, 0],
            "body_valid": valid[:, 1],
        }

    @staticmethod
    def select_feature(fused_feature, parts, mode):
        if mode == "fused":
            return fused_feature
        if mode == "global":
            return parts["global"]
        raise ValueError("Unsupported CaB-ReID feature mode: {!r}".format(mode))
