from dataclasses import dataclass


@dataclass(frozen=True)
class CaBReIDConfig:
    mask_threshold: float = 0.45
    binary_pooling: bool = True
    min_part_ratio: float = 0.075
    part_validity_enabled: bool = False
    cab_invalid_viewids: tuple = ()
    body_exclusive_from_cab: bool = True
    body_valid_area_minus_cab: bool = True
    pool_beta: float = 0.0
    global_weight: float = 1.0
    cab_weight: float = 0.5
    body_weight: float = 0.5
    mask_dropout: float = 0.0
    online_mask: bool = True

    @classmethod
    def from_yacs(cls, cfg):
        part = cfg.PART_MEMORY
        return cls(
            mask_threshold=float(part.MASK_THRESHOLD),
            binary_pooling=bool(part.BINARY_POOLING),
            min_part_ratio=float(part.MIN_PART_RATIO),
            part_validity_enabled=bool(part.PART_VALIDITY_ENABLED),
            cab_invalid_viewids=tuple(int(value) for value in part.CAB_INVALID_VIEWIDS),
            body_exclusive_from_cab=bool(part.BODY_EXCLUSIVE_FROM_CAB),
            body_valid_area_minus_cab=bool(part.BODY_VALID_AREA_MINUS_CAB),
            pool_beta=float(part.POOL_BETA),
            global_weight=float(part.GLOBAL_WEIGHT),
            cab_weight=float(part.CAB_WEIGHT),
            body_weight=float(part.BODY_WEIGHT),
            mask_dropout=float(part.MASK_DROPOUT),
            online_mask=bool(part.ONLINE_MASK),
        )
