"""Portable CaB-ReID modules shared by all backbone adapters."""

from .config import CaBReIDConfig
from .masking import PromptRegionMasker, RegionAdapter
from .module import CaBReIDModule

__all__ = ["CaBReIDConfig", "CaBReIDModule", "PromptRegionMasker", "RegionAdapter"]
