import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionAdapter(nn.Module):
    def __init__(self, dim, init_logit_scale=10.0):
        super().__init__()
        self.text_proj = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.text_proj.weight)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(init_logit_scale), dtype=torch.float32))

    def forward(self, patch_features, text_features):
        adapted_text = F.normalize(self.text_proj(text_features), dim=-1)
        patch_features = patch_features.to(dtype=adapted_text.dtype)
        logits = patch_features @ adapted_text.t()
        return self.logit_scale.exp().clamp(max=100.0) * logits


class PromptRegionMasker(nn.Module):
    """Frozen prompt-guided Cab/Body mask generator for any token backbone."""

    def __init__(self, encoder, checkpoint_path, token_hw, reid_mean, reid_std):
        super().__init__()
        self.encoder = encoder
        self.token_hw = tuple(int(value) for value in token_hw)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "adapter_state_dict" not in checkpoint or "text_features" not in checkpoint:
            raise KeyError("Region checkpoint must contain adapter_state_dict and text_features")
        text_features = checkpoint["text_features"].float()
        self.adapter = RegionAdapter(dim=int(text_features.shape[-1]))
        self.adapter.load_state_dict(checkpoint["adapter_state_dict"])
        for parameter in self.adapter.parameters():
            parameter.requires_grad_(False)

        self.register_buffer("text_features", F.normalize(text_features, dim=-1), persistent=False)
        self.register_buffer("reid_mean", torch.tensor(reid_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("reid_std", torch.tensor(reid_std).view(1, 3, 1, 1), persistent=False)
        self.register_buffer(
            "clip_mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "clip_std", torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1), persistent=False
        )

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, image):
        if image is None:
            raise ValueError("PromptRegionMasker requires an image tensor")
        self.encoder.eval()
        self.adapter.eval()
        with torch.no_grad():
            mean = self.reid_mean.to(device=image.device, dtype=image.dtype)
            std = self.reid_std.to(device=image.device, dtype=image.dtype)
            rgb = (image * std + mean).clamp(0.0, 1.0)
            clip_mean = self.clip_mean.to(device=image.device, dtype=image.dtype)
            clip_std = self.clip_std.to(device=image.device, dtype=image.dtype)
            encoder_input = (rgb - clip_mean) / clip_std
            encoder_dtype = next(self.encoder.parameters()).dtype
            _last, _tokens, patch_features = self.encoder(encoder_input.to(dtype=encoder_dtype))
            patch_features = F.normalize(patch_features[:, 1:, :].float(), dim=-1)
            token_h, token_w = self.token_hw
            if patch_features.shape[1] != token_h * token_w:
                raise RuntimeError(
                    "Region token grid mismatch: got {}, expected {}x{}".format(
                        patch_features.shape[1], token_h, token_w
                    )
                )
            logits = self.adapter(
                patch_features,
                self.text_features.to(device=patch_features.device, dtype=patch_features.dtype),
            )
            return torch.sigmoid(logits).transpose(1, 2).reshape(image.shape[0], 2, token_h, token_w)
