from collections import OrderedDict
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from cabreid.checkpoint import EVALUATION_FORMAT, SHARED_ENCODER, load_checkpoint, sha256
from cabreid.evaluation import EvaluationMode


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "model.pth"
        self.model = nn.Module()
        self.model.classifier = nn.Linear(2, 3, bias=False)
        self.model.cabreid = nn.Module()
        masker = nn.Module()
        masker.encoder = nn.Linear(2, 2, bias=False)
        masker.adapter = nn.Linear(2, 2, bias=False)
        masker.register_buffer("text_features", torch.zeros(2, 2), persistent=False)
        self.model.cabreid.region_masker = masker

    def test_native_names_and_data_parallel(self):
        state = OrderedDict([
            ("module.classifier.weight", torch.ones(3, 2)),
            ("module.online_part_encoder.weight", torch.full((2, 2), 2.0)),
            ("module.online_region_adapter.weight", torch.full((2, 2), 3.0)),
            ("module.online_text_features", torch.full((2, 2), 4.0)),
        ])
        torch.save(state, self.path)
        load_checkpoint(self.model, self.path)
        self.assertTrue(torch.equal(self.model.classifier.weight, torch.ones(3, 2)))
        masker = self.model.cabreid.region_masker
        self.assertTrue(torch.equal(masker.encoder.weight, torch.full((2, 2), 2.0)))
        self.assertTrue(torch.equal(masker.adapter.weight, torch.full((2, 2), 3.0)))
        self.assertTrue(torch.equal(masker.text_features, torch.full((2, 2), 4.0)))

    def test_portable_state(self):
        torch.save({"state_dict": self.model.state_dict()}, self.path)
        load_checkpoint(self.model, self.path)

    def test_missing_parameter_is_rejected(self):
        state = self.model.state_dict()
        del state["classifier.weight"]
        torch.save(state, self.path)
        with self.assertRaises(RuntimeError):
            load_checkpoint(self.model, self.path)

    def test_unexpected_parameter_is_rejected(self):
        state = self.model.state_dict()
        state["unrecognised.weight"] = torch.zeros(1)
        torch.save(state, self.path)
        with self.assertRaises(RuntimeError):
            load_checkpoint(self.model, self.path)

    def compact_bundle(self):
        del self.model.classifier
        self.model.evaluation_spec = {"backbone": "clipreid", "dataset": "mvti",
                                      "image_size": [256, 128], "stride": [12, 12]}
        state = self.model.state_dict()
        key = "cabreid.region_masker.encoder.weight"
        shared_path = self.path.parent / SHARED_ENCODER
        torch.save({"format": "cabreid-shared-encoder", "state_dict": {key: state.pop(key)}}, shared_path)
        return {
            "format": EVALUATION_FORMAT, "model": self.model.evaluation_spec.copy(),
            "shared_encoder": SHARED_ENCODER, "shared_sha256": sha256(shared_path),
            "state_dict": state,
            "buffers": {"cabreid.region_masker.text_features": torch.full((2, 2), 4.0)},
        }

    def test_evaluation_bundle(self):
        bundle = self.compact_bundle()
        torch.save(bundle, self.path)
        load_checkpoint(self.model, self.path)
        self.assertTrue(torch.equal(self.model.cabreid.region_masker.text_features, torch.full((2, 2), 4.0)))

    def test_evaluation_bundle_rejects_training_model(self):
        bundle = self.compact_bundle()
        del self.model.evaluation_spec
        torch.save(bundle, self.path)
        with self.assertRaisesRegex(RuntimeError, "evaluation_only=True"):
            load_checkpoint(self.model, self.path)

    def test_evaluation_bundle_rejects_wrong_dataset(self):
        bundle = self.compact_bundle()
        bundle["model"]["dataset"] = "trc31k"
        torch.save(bundle, self.path)
        with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
            load_checkpoint(self.model, self.path)

    def test_evaluation_bundle_requires_shared_file(self):
        bundle = self.compact_bundle()
        torch.save(bundle, self.path)
        (self.path.parent / SHARED_ENCODER).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "shared encoder"):
            load_checkpoint(self.model, self.path)

    def test_evaluation_bundle_checks_shared_hash(self):
        bundle = self.compact_bundle()
        bundle["shared_sha256"] = "0" * 64
        torch.save(bundle, self.path)
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            load_checkpoint(self.model, self.path)

    def test_evaluation_bundle_rejects_path_traversal(self):
        bundle = self.compact_bundle()
        bundle["shared_encoder"] = "../encoder.pth"
        torch.save(bundle, self.path)
        with self.assertRaisesRegex(ValueError, "filename"):
            load_checkpoint(self.model, self.path)

    def test_evaluation_bundle_rejects_overlap(self):
        bundle = self.compact_bundle()
        bundle["state_dict"]["cabreid.region_masker.encoder.weight"] = torch.zeros(2, 2)
        torch.save(bundle, self.path)
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            load_checkpoint(self.model, self.path)

    def test_evaluation_bundle_still_requires_every_inference_parameter(self):
        bundle = self.compact_bundle()
        bundle["state_dict"].clear()
        torch.save(bundle, self.path)
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            load_checkpoint(self.model, self.path)

    def test_evaluation_bundle_requires_buffers(self):
        bundle = self.compact_bundle()
        bundle["buffers"].clear()
        torch.save(bundle, self.path)
        with self.assertRaisesRegex(RuntimeError, "mask buffers"):
            load_checkpoint(self.model, self.path)

    def test_evaluation_bundle_rejects_precision_changes(self):
        bundle = self.compact_bundle()
        key = "cabreid.region_masker.adapter.weight"
        bundle["state_dict"][key] = bundle["state_dict"][key].half()
        torch.save(bundle, self.path)
        with self.assertRaisesRegex(RuntimeError, "dtype mismatch"):
            load_checkpoint(self.model, self.path)

    def test_evaluation_mode_blocks_training(self):
        class Model(EvaluationMode, nn.Module):
            pass

        model = Model()
        model.train()
        model.evaluation_spec = {"backbone": "clipreid"}
        model.eval()
        with self.assertRaisesRegex(RuntimeError, "evaluation-only"):
            model.train()
