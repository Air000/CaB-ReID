import unittest

import torch

from cabreid.config import CaBReIDConfig
from cabreid.pooling import MaskedTokenPool, PartFeatureFusion, PartMaskProcessor


class PortableModuleEquivalenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = CaBReIDConfig(
            mask_threshold=0.45,
            binary_pooling=True,
            part_validity_enabled=False,
            body_exclusive_from_cab=True,
            body_valid_area_minus_cab=True,
            global_weight=1.0,
            cab_weight=0.5,
            body_weight=0.5,
        )

    def test_exclusive_masks_match_v4_0_formula(self):
        masks = torch.rand(3, 2, 4, 5)
        expected_cab = (masks[:, 0:1] >= 0.45).float()
        expected_body = (masks[:, 1:2] >= 0.45).float() * (1.0 - expected_cab)
        actual, valid = PartMaskProcessor(self.config)(masks)
        self.assertTrue(torch.equal(actual[:, 0:1], expected_cab))
        self.assertTrue(torch.equal(actual[:, 1:2], expected_body))
        self.assertTrue(valid.all())

    def test_masked_pool_matches_v4_0_formula(self):
        tokens = torch.randn(3, 21, 8)
        mask = torch.randint(0, 2, (3, 20)).float()
        denominator = mask.sum(1, keepdim=True)
        expected = (tokens[:, 1:] * mask.unsqueeze(-1)).sum(1) / denominator.clamp_min(1e-6)
        has_region = (denominator > 1e-6).float()
        expected = expected * has_region + tokens[:, 0] * (1.0 - has_region)
        actual = MaskedTokenPool()(tokens, mask)
        self.assertTrue(torch.equal(actual, expected))

    def test_zero_threshold_keeps_soft_masks(self):
        config = CaBReIDConfig(mask_threshold=0.0, binary_pooling=True)
        masks = torch.rand(2, 2, 3, 4)
        actual, _valid = PartMaskProcessor(config)(masks)
        expected = masks.clone()
        expected[:, 1:2] = (masks[:, 1:2] > 0).float() * (1.0 - (masks[:, 0:1] > 0).float())
        self.assertTrue(torch.equal(actual, expected))

    def test_fusion_matches_v4_0_formula(self):
        global_feature = torch.randn(3, 8)
        cab_feature = torch.randn(3, 8)
        body_feature = torch.randn(3, 8)
        valid = torch.tensor(((True, True), (True, False), (False, True)))
        cab_weight = valid[:, 0].float().view(-1, 1) * 0.5
        body_weight = valid[:, 1].float().view(-1, 1) * 0.5
        expected = (
            global_feature + cab_feature * cab_weight + body_feature * body_weight
        ) / (1.0 + cab_weight + body_weight)
        actual = PartFeatureFusion(1.0, 0.5, 0.5)(
            global_feature, cab_feature, body_feature, valid
        )
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
