import torch
import torch.nn as nn
import numpy as np
from cabreid import CaBReIDConfig, CaBReIDModule, PromptRegionMasker
from cabreid.checkpoint import load_checkpoint
from cabreid.evaluation import EvaluationMode, prepare_evaluation_model, visual_encoder
from .clip.model import VisionTransformer
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts): 
        x = prompts + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  # NLD -> LND 
        x = self.transformer(x) 
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) 

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection 
        return x

class build_transformer(EvaluationMode, nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, evaluation_only=False):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE   

        if not evaluation_only:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
            self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        if evaluation_only:
            self.image_encoder = visual_encoder(
                VisionTransformer, self.model_name,
                (self.h_resolution, self.w_resolution), self.vision_stride_size,
            )
        else:
            clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
            clip_model.to("cuda")
            self.image_encoder = clip_model.visual

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        if not evaluation_only:
            dataset_name = cfg.DATASETS.NAMES
            self.prompt_learner = PromptLearner(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
            self.text_encoder = TextEncoder(clip_model)
        self.part_memory_enabled = bool(cfg.PART_MEMORY.ENABLED)
        self.part_memory_train_main_feature = str(cfg.PART_MEMORY.TRAIN_MAIN_FEATURE).lower()
        self.part_memory_memory_feature = str(cfg.PART_MEMORY.MEMORY_FEATURE).lower()
        self.part_memory_test_feature = str(cfg.PART_MEMORY.TEST_FEATURE).lower()
        for name, value in (
            ("PART_MEMORY.TRAIN_MAIN_FEATURE", self.part_memory_train_main_feature),
            ("PART_MEMORY.MEMORY_FEATURE", self.part_memory_memory_feature),
            ("PART_MEMORY.TEST_FEATURE", self.part_memory_test_feature),
        ):
            if value not in ("fused", "global"):
                raise ValueError(f"{name} must be 'fused' or 'global', got {value!r}")
        cab_config = CaBReIDConfig.from_yacs(cfg)
        region_masker = None
        if cab_config.online_mask:
            if evaluation_only:
                region_encoder = visual_encoder(
                    VisionTransformer, self.model_name,
                    (self.h_resolution, self.w_resolution), self.vision_stride_size,
                )
            else:
                online_clip_model = load_clip_to_cpu(
                    self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size
                )
                online_clip_model.to("cuda")
                region_encoder = online_clip_model.visual
            region_masker = PromptRegionMasker(
                region_encoder,
                cfg.PART_MEMORY.ONLINE_ADAPTER_PATH,
                (self.h_resolution, self.w_resolution),
                cfg.INPUT.PIXEL_MEAN,
                cfg.INPUT.PIXEL_STD,
            )
        self.cabreid = CaBReIDModule(cab_config, region_masker)

    def _neck_feature(self, img_feature, img_feature_proj):
        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)
        return torch.cat([feat, feat_proj], dim=1)

    def _part_memory_features(self, image_features, image_features_proj, part_masks, online_image=None, part_view_label=None):
        global_feat = self._neck_feature(image_features[:, 0], image_features_proj[:, 0])
        return self.cabreid(
            (image_features, image_features_proj),
            global_feat,
            masks=part_masks,
            image=online_image,
            view_labels=part_view_label,
            token_hw=(self.h_resolution, self.w_resolution),
            projector=lambda features: self._neck_feature(features[0], features[1]),
        )

    def _select_part_feature(self, fused_feat, part_dict, mode):
        return self.cabreid.select_feature(fused_feat, part_dict, mode)

    def forward(self, x = None, label=None, get_image = False, get_text = False, cam_label= None, view_label=None, mask=None, part_view_label=None, return_part_features=False, feature_mode=None):
        if get_text == True:
            if getattr(self, "evaluation_spec", None) is not None:
                raise RuntimeError("Text-prompt training is unavailable in an evaluation-only model.")
            prompts = self.prompt_learner(label) 
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            return text_features

        if get_image == True:
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:,0]
        
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed) 
            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]

        feat = self.bottleneck(img_feature) 
        feat_proj = self.bottleneck_proj(img_feature_proj) 
        
        part_dict = None
        fused_part_feat = None
        if self.part_memory_enabled:
            fused_part_feat, part_dict = self._part_memory_features(
                image_features,
                image_features_proj,
                mask,
                online_image=x,
                part_view_label=part_view_label,
            )

        out_part_feat = None
        if fused_part_feat is not None and part_dict is not None:
            if self.training:
                selected_mode = self.part_memory_train_main_feature
            elif feature_mode == "memory":
                selected_mode = self.part_memory_memory_feature
            else:
                selected_mode = self.part_memory_test_feature
            out_part_feat = self._select_part_feature(fused_part_feat, part_dict, selected_mode)

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            if return_part_features:
                main_feat = out_part_feat if out_part_feat is not None else torch.cat([feat, feat_proj], dim=1)
                return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj, main_feat, part_dict
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj

        else:
            if self.neck_feat == 'after':
                out = torch.cat([feat, feat_proj], dim=1)
            else:
                out = torch.cat([img_feature, img_feature_proj], dim=1)
            if out_part_feat is not None:
                out = out_part_feat
            if return_part_features:
                return out, part_dict
            return out


    def load_param(self, trained_path):
        load_checkpoint(self, trained_path)

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num, evaluation_only=False):
    model = build_transformer(num_class, camera_num, view_num, cfg, evaluation_only=evaluation_only)
    return prepare_evaluation_model(model, "clipreid", cfg) if evaluation_only else model


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model

class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name in ["VehicleID", "veri", "trc31k", "mvti_single"]:
            ctx_init = "A photo of a X X X X vehicle."
        else:
            ctx_init = "A photo of a X X X X person."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4
        
        tokenized_prompts = clip.tokenize(ctx_init).cuda() 
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype) 
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype) 
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors) 

        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])  
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx: , :])  
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx = self.cls_ctx[label] 
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1) 
        suffix = self.token_suffix.expand(b, -1, -1) 
            
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        ) 

        return prompts 
