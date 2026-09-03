# CaB-ReID v4.1 and TRC-31K

CaB-ReID is a truck re-identification method that models the cab and body as complementary identity cues. TRC-31K is the image-level truck Re-ID benchmark used for its training and evaluation protocol.

This repository is the project page for the code, dataset protocol, and reproducibility material developed as part of Shihan Xu's research at Swinburne University of Technology.

> **Release status:** public-release candidate. Dataset access, project licensing, institutional approval, final privacy clearance, and public-pixel checkpoint validation must be completed before the repository is announced.

## Why truck Re-ID?

Truck re-identification matches the same vehicle across non-overlapping cameras. It is difficult because viewpoint, scale, illumination, occlusion, articulated body configuration, and loading-state changes can substantially alter a truck's appearance. CaB-ReID addresses this by using prompt-guided cab and body regions, region-aware pooling, fused retrieval descriptors, and separate CabMem/BodyMem identity memories.

## Dataset examples

The following privacy-curated TRC-31K samples illustrate the range of truck body configurations and viewpoints. Any manually confirmed registration plate or real-person face in these examples is blurred.

| Box truck | Curtainside truck | Concrete truck |
|---|---|---|
| <img src="assets/trc31k_examples/example_box_truck.png" width="280" alt="Privacy-curated TRC-31K box truck sample"> | <img src="assets/trc31k_examples/example_curtainside_truck.png" width="280" alt="Privacy-curated TRC-31K curtainside truck sample"> | <img src="assets/trc31k_examples/example_concrete_truck.png" width="280" alt="Privacy-curated TRC-31K concrete truck sample"> |

## Contributions

- **TRC-31K v1.0:** an image-level truck Re-ID protocol with 31,619 images, 1,305 identities, and five cameras.
- **CaB-ReID:** a cab/body-aware representation using prompt-guided region masks, exclusive body pooling, feature fusion, and part-specific identity memories.
- **Portable implementation:** one shared CaB-ReID module package used by CLIP-ReID and TransReID adapters.
- **Reproducibility:** authoritative split manifests, file checksums, release validators, training configurations, and step-by-step commands.

## Releases

| Release | Purpose | Location | Status |
|---|---|---|---|
| TRC-31K v1.0 | Train/query/gallery benchmark | Separate dataset archive | Release candidate |
| CaB-ReID v4.1 | Portable method with unified backbone CLI | This repository | Public-pixel validation pending |
| CaB-ReID v4.0 | Original code corresponding to the reported experiments | Local reference archive, outside this repository | Preserved reference |
| Trained checkpoints | Direct evaluation and benchmarking | To be published after public-pixel validation | Pending |

The dataset archive will be hosted separately because of its size. The public download URL and archive checksum will be added here after release approval. This repository contains the code and project-page assets. Dataset images, CSV manifests, checksum files, and the dataset validator are distributed in the separate archive.

## TRC-31K protocol

| Split | Images | Identities |
|---|---:|---:|
| Train | 25,941 | 1,073 |
| Query | 2,292 | 232 |
| Gallery | 3,386 | 232 |
| **Total** | **31,619** | **1,305 unique** |

Training and test identities are disjoint. Query and gallery contain the same 232 test identities. Evaluation excludes gallery images sharing both identity and camera with the query, then reports mAP and CMC Rank-1, Rank-5, and Rank-10. The dataset archive includes its own README and authoritative split CSVs. To preserve the reported training protocol, it retains 16 exact-pixel repetition groups: two same-PID and 14 legacy cross-PID groups, with two training rows per group. Query and gallery contain no exact-pixel repetitions.

## Reference results

| Backbone | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| CLIP-ReID + CaB-ReID v4.0 | 87.0 | 94.5 | 97.6 | 98.9 |
| TransReID + CaB-ReID v4.0 | 87.2 | 94.5 | 97.1 | 98.6 |

These reference values were produced from the private pre-anonymisation experiment archive. The public release preserves the identities and split protocol but contains privacy-curated pixels. Fresh v4.1 results and trained checkpoints will be published after both backbones have been rerun on the public images.

## Layout

```text
CaB-ReID/
├── assets/                # project-page images
├── cabreid/               # portable method modules
├── clipreid/              # CLIP-ReID backbone adapter and native trainer
├── transreid/             # TransReID backbone adapter and native trainer
├── weights/
├── main.py                # unified CLI
├── requirements.txt
└── CHECKSUMS.sha256
```

## Environment

Use Python 3.10 or 3.11 with CUDA-enabled PyTorch and torchvision. In Colab:

```bash
git clone https://github.com/Air000/CaB-ReID.git
cd CaB-ReID
python -m pip install -r requirements.txt
```

Download the ImageNet-pretrained ViT-B/16 checkpoint required by TransReID:

```bash
wget -O weights/jx_vit_base_p16_224-80ecf9dd.pth \
  https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
```

Both backbones use `weights/clip_region_adapter.pt`. CLIP ViT-B/16 is downloaded automatically on first use.

## Dataset

The sibling dataset must have this layout:

```text
../dataset/TRC_31K_v1.0/train/
../dataset/TRC_31K_v1.0/query/
../dataset/TRC_31K_v1.0/gallery/
../dataset/TRC_31K_v1.0/metadata/{train,query,gallery}.csv
```

Validate it before training:

```bash
python tools/validate_release.py
```

Validate the source layout (no v4.0 checkout is required):

```bash
python tools/validate_code.py
```

For maintainers with the preserved local v4.0 package, also check configuration parity:

```bash
python tools/validate_code.py --reference-code ../code
```

## Unified training

Select the backbone on the command line:

```bash
python main.py train --backbone clipreid
python main.py train --backbone transreid
```

YACS overrides follow `--`:

```bash
python main.py train --backbone clipreid -- \
  SOLVER.STAGE1.IMS_PER_BATCH 16 \
  SOLVER.STAGE2.IMS_PER_BATCH 16 \
  TEST.IMS_PER_BATCH 16

python main.py train --backbone transreid -- \
  SOLVER.IMS_PER_BATCH 16 \
  TEST.IMS_PER_BATCH 16
```

CLIP-ReID retains its 60-epoch prompt-learning stage and 60-epoch Re-ID stage. TransReID retains its native 120-epoch schedule.

## Unified evaluation

```bash
python main.py evaluate --backbone clipreid \
  --weight outputs/v4_1_clipreid/ViT-B-16_60.pth

python main.py evaluate --backbone transreid \
  --weight outputs/v4_1_transreid/transformer_120.pth
```

Evaluation reports mAP and CMC Rank-1, Rank-5, and Rank-10 using the TRC-31K query/gallery protocol.

## Portable method interface

The method is implemented once in the shared package:

| Module | Responsibility | Backbone assumption |
|---|---|---|
| `cabreid.masking.PromptRegionMasker` | Frozen prompt-guided Cab/Body masks | Encoder returns projected patch tokens |
| `cabreid.pooling.PartMaskProcessor` | Thresholding, exclusive Body mask, validity | None |
| `cabreid.pooling.MaskedTokenPool` | Mask-weighted token pooling with CLS fallback | Tokens use `[CLS, patches...]` |
| `cabreid.pooling.PartFeatureFusion` | Weighted Global/Cab/Body fusion | Features share one descriptor space |
| `cabreid.module.CaBReIDModule` | Masking, pooling, projection, and fusion | Adapter supplies tokens and global feature |
| `cabreid.memory.ClusterMemory` | CabMem/BodyMem identity prototypes | None |
| `cabreid.data` | Spatially aligned image/mask transforms | None |

A backbone adapter supplies:

1. `token_streams`: one tensor or a tuple shaped `[batch, 1 + patches, channels]`.
2. `global_feature`: the native global descriptor used as the fusion anchor.
3. `token_hw`: the patch-grid height and width.
4. An optional `projector(parts)` when pooled tokens must be mapped into the global descriptor space.

```python
fused, parts = cabreid(
    token_streams,
    global_feature,
    masks=masks,
    image=image,
    view_labels=view_labels,
    token_hw=token_hw,
    projector=projector,
)
```

The call returns the fused descriptor plus Global, Cab, Body, Cab-valid, and Body-valid outputs. CLIP-ReID passes its native and projected token streams, then applies both necks in its projector. TransReID passes the JPM global token stream directly and needs no projector. Backbone-native classification, metric losses, local branches, optimizer, and schedule remain outside the portable method.

To add another ViT-style Re-ID backbone, expose its global and patch tokens, instantiate `CaBReIDModule` from the shared configuration, and use the call above. CabMem/BodyMem consumes `parts["cab"]`, `parts["body"]`, and their validity flags without backbone-specific memory code.

## Benchmarking and reuse

Researchers can use the published query/gallery protocol to compare other truck Re-ID methods directly with CaB-ReID. The CSV manifests are authoritative: each row records the relative image path, identity, and zero-based camera index. Please report the exact release version, backbone, image size, checkpoint, and whether reranking was used.

The v4.1 `cabreid` package is separated from the backbone implementations so that its masking, pooling, fusion, and memory components can be integrated into other token-based Re-ID models.

## Privacy and responsible use

The release images were reviewed for visible registration plates and real-person faces. The public pixels contain strong blur over 109 manually verified regions in 97 images. Pre-anonymisation images, reviewer records, curation backups, and intermediate audit material are not distributed.

TRC-31K is intended for academic research on vehicle re-identification and representation learning. Users are responsible for complying with the final dataset licence, applicable privacy requirements, and institutional policies.

## Citation

The associated journal manuscript has not yet received a permanent bibliographic record. A verified BibTeX entry will be added after publication. Until then, please identify the resource as **CaB-ReID with TRC-31K v1.0** and link to this project page.

## Licence and access

The upstream CLIP-ReID and TransReID licences are retained inside their respective source directories. The CaB-ReID project licence and TRC-31K dataset licence are pending approval. Do not redistribute this release candidate until those terms and the public download notice are added.

## Contact

Research lead: **Shihan Xu**, Swinburne University of Technology. A public project contact address will be added before release.
