"""Evaluation-only model construction."""


TRAINING_MODULES = {
    "clipreid": ("classifier", "classifier_proj", "prompt_learner", "text_encoder"),
    "transreid": ("classifier", "classifier_1", "classifier_2", "classifier_3", "classifier_4"),
}


class EvaluationMode:
    def train(self, mode=True):
        if mode and getattr(self, "evaluation_spec", None) is not None:
            raise RuntimeError("This model is evaluation-only; construct a training model to train.")
        return super().train(mode)


def visual_encoder(vision_class, name, token_hw, stride):
    if name != "ViT-B-16":
        raise ValueError("Evaluation-only construction supports CLIP ViT-B-16.")
    return vision_class(
        h_resolution=token_hw[0], w_resolution=token_hw[1], patch_size=16,
        stride_size=stride, width=768, layers=12, heads=12, output_dim=512,
    )


def prepare_evaluation_model(model, backbone, cfg):
    for name in TRAINING_MODULES[backbone]:
        if hasattr(model, name):
            delattr(model, name)
    model.evaluation_spec = {
        "backbone": backbone,
        "dataset": {"mvti_single": "mvti", "trc31k": "trc31k"}[cfg.DATASETS.NAMES],
        "image_size": list(cfg.INPUT.SIZE_TRAIN),
        "stride": list(cfg.MODEL.STRIDE_SIZE),
    }
    return model.eval()
