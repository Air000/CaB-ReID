# CaB-ReID: Cab-and-Body Prompt-Guided Truck Re-Identification

CaB-ReID is a truck re-identification method that models the cab and body as complementary identity cues. TRC-31K is the image-level truck Re-ID benchmark used for its training and evaluation protocol.

**TRC-31K will be released once the paper is accepted.** This repository provides the CaB-ReID implementation and instructions for evaluating the final paper checkpoints with CLIP-ReID and TransReID backbones.

**[Download evaluation checkpoints](https://github.com/Air000/CaB-ReID/releases/tag/checkpoints)** (pre-release). See [checkpoint setup](#3-prepare-and-verify-the-checkpoints) for the shared files required by both backbones.

## Why truck Re-ID?

Truck re-identification matches the same vehicle across non-overlapping cameras. It is difficult because viewpoint, scale, illumination, occlusion, articulated body configuration, and loading-state changes can substantially alter a truck's appearance. CaB-ReID addresses this by using prompt-guided cab and body regions, region-aware pooling, fused retrieval descriptors, and separate CabMem/BodyMem identity memories.

## Dataset examples

The following privacy-curated TRC-31K samples illustrate the range of truck body configurations and viewpoints. Any manually confirmed registration plate or real-person face in these examples is blurred.

| Box truck | Curtainside truck | Concrete truck |
|---|---|---|
| <img src="assets/trc31k_examples/example_box_truck.png" width="280" alt="Privacy-curated TRC-31K box truck sample"> | <img src="assets/trc31k_examples/example_curtainside_truck.png" width="280" alt="Privacy-curated TRC-31K curtainside truck sample"> | <img src="assets/trc31k_examples/example_concrete_truck.png" width="280" alt="Privacy-curated TRC-31K concrete truck sample"> |

## Contributions

- **TRC-31K:** an image-level truck Re-ID protocol with 31,619 images, 1,305 identities, and five cameras.
- **CaB-ReID:** a cab/body-aware representation using prompt-guided region masks, exclusive body pooling, feature fusion, and part-specific identity memories.
- **Portable implementation:** one shared CaB-ReID module package used by CLIP-ReID and TransReID adapters.
- **Reproducibility:** authoritative split manifests, file checksums, release validators, training configurations, and step-by-step commands.

## Resources

| Resource | Purpose | Location | Status |
|---|---|---|---|
| TRC-31K | Train/query/gallery benchmark | Separate dataset archive | Release after paper acceptance |
| MV-TI | Single-view combined-gallery benchmark | [Official dataset page](https://github.com/maybeextra/DAG-UMB) | Public dataset |
| CaB-ReID | Portable method with unified backbone CLI | This repository | Source code |
| Evaluation checkpoints | Final paper models on both datasets | [GitHub release assets](https://github.com/Air000/CaB-ReID/releases/tag/checkpoints) | Available as a pre-release |

TRC-31K images, split manifests, checksums, and its validator will be distributed in a separate archive after paper acceptance. No dataset images other than the examples shown here are included in the source repository.

## TRC-31K protocol

| Split | Images | Identities |
|---|---:|---:|
| Train | 25,941 | 1,073 |
| Query | 2,292 | 232 |
| Gallery | 3,386 | 232 |
| **Total** | **31,619** | **1,305 unique** |

Training and test identities are disjoint. Query and gallery contain the same 232 test identities. Evaluation excludes gallery images sharing both identity and camera with the query, then reports mAP and CMC Rank-1, Rank-5, and Rank-10. The dataset archive includes its own README and authoritative split CSVs. To preserve the reported training protocol, it retains 16 exact-pixel repetition groups: two same-PID and 14 legacy cross-PID groups, with two training rows per group. Query and gallery contain no exact-pixel repetitions.

## Final checkpoint results

The following results use the final full-training checkpoints, not intermediate or best-epoch selections. All metrics are percentages; reranking is disabled.

| Dataset | Backbone | Checkpoint in `weights/` | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---|---|---:|---:|---:|---:|
| MV-TI | CLIP-ReID + CaB-ReID | `cabreid_clipreid_mvti.pth` | 89.3 | 94.6 | 96.8 | 97.6 |
| MV-TI | TransReID + CaB-ReID | `cabreid_transreid_mvti.pth` | 92.6 | 96.1 | 97.2 | 97.6 |
| TRC-31K | CLIP-ReID + CaB-ReID | `cabreid_clipreid_trc31k.pth` | 87.0 | 94.5 | 97.6 | 98.9 |
| TRC-31K | TransReID + CaB-ReID | `cabreid_transreid_trc31k.pth` | 87.2 | 94.5 | 97.1 | 98.6 |

CLIP-ReID checkpoints are from the final epoch of the 60-epoch second stage; TransReID checkpoints are from epoch 120. The evaluation files retain the original FP32 inference tensors without quantisation. Training-only classifiers and CLIP text/prompt components are omitted, and the frozen region encoder is stored once for all four models. Each model retains its dataset-specific positional embeddings. The loader verifies the shared encoder checksum and rejects configuration mismatches or missing/unexpected inference parameters.

The TRC-31K paper results were measured on the pre-anonymisation experiment images. The release preserves the identities and split protocol but contains privacy-curated pixels, so these exact scores are not yet verified on the release images. Full-dataset evaluation with the portable implementation remains to be confirmed; the table records the original experiment results.

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

## Checkpoint evaluation

### 1. Set up the environment

Use a CUDA GPU. In Colab, select a GPU runtime before running the commands below; its PyTorch and torchvision installation can be used directly.

```bash
git clone https://github.com/Air000/CaB-ReID.git
cd CaB-ReID
python -m pip install -r requirements.txt
python -c "import torch; assert torch.cuda.is_available(), 'Select a GPU runtime'"
```

For a separate machine, use Python 3.10 or newer and install CUDA-enabled PyTorch and matching torchvision first. Run every command below from the repository root.

### 2. Prepare the dataset

**MV-TI:** obtain the dataset from its [official project page](https://github.com/maybeextra/DAG-UMB). Use the original split-name lists and the individual front, side, and back images:

```text
../dataset/MV-TI/
├── train_names.txt
├── query_names.txt
├── gallery_names.txt
├── front/{train,query,gallery}/
├── side/{train,query,gallery}/
└── back/{train,query,gallery}/
```

Each listed sample expands in Front/Side/Back order to three images named `<sample>_F.jpg`, `<sample>_S.jpg`, and `<sample>_B.jpg`. The paper evaluates the individual views in one combined gallery, not the concatenated FS/FB/SB/FSB images. The expected protocol has 1,776 training identities, 2,421 query images, and 18,201 gallery images. Do not resplit the data.

**TRC-31K:** the dataset will be released once the paper is accepted. The commands below are for users with authorised access now or for use after that release:

```text
../dataset/TRC-31K/
├── train/
├── query/
├── gallery/
└── metadata/{train,query,gallery}.csv
```

Keep all three split manifests and image directories, including the training split: the dataset loader still validates and enumerates the complete benchmark protocol. Evaluation-only models do not include identity classification heads. The folder name is not fixed. Validate the extracted TRC-31K archive with:

```bash
python tools/validate_release.py ../dataset/TRC-31K
```

### 3. Prepare and verify the checkpoints

Download the chosen dataset/backbone checkpoint into `weights/`. Every model also requires the shared region encoder and the small prompt-region adapter in that folder. Keep the checksum manifest there for verification. The other three model checkpoints are optional.

| Download | Purpose | Size |
|---|---|---:|
| [cabreid_clipreid_mvti.pth](https://github.com/Air000/CaB-ReID/releases/download/checkpoints/cabreid_clipreid_mvti.pth) | CLIP-ReID + CaB-ReID on MV-TI | 347 MB |
| [cabreid_transreid_mvti.pth](https://github.com/Air000/CaB-ReID/releases/download/checkpoints/cabreid_transreid_mvti.pth) | TransReID + CaB-ReID on MV-TI | 405 MB |
| [cabreid_clipreid_trc31k.pth](https://github.com/Air000/CaB-ReID/releases/download/checkpoints/cabreid_clipreid_trc31k.pth) | CLIP-ReID + CaB-ReID on TRC-31K | 348 MB |
| [cabreid_transreid_trc31k.pth](https://github.com/Air000/CaB-ReID/releases/download/checkpoints/cabreid_transreid_trc31k.pth) | TransReID + CaB-ReID on TRC-31K | 406 MB |
| [cabreid_region_encoder.pth](https://github.com/Air000/CaB-ReID/releases/download/checkpoints/cabreid_region_encoder.pth) | Shared encoder; required by every model | 344 MB |
| [clip_region_adapter.pt](https://github.com/Air000/CaB-ReID/releases/download/checkpoints/clip_region_adapter.pt) | Shared adapter; required by every model | 1 MB |
| [CHECKSUMS.sha256](https://github.com/Air000/CaB-ReID/releases/download/checkpoints/CHECKSUMS.sha256) | SHA-256 verification manifest | <1 KB |

The complete evaluation bundle is approximately **1.85 GB**, compared with 3.27 GB for the full checkpoints. Each CLIP-ReID checkpoint is 347–348 MB; each TransReID checkpoint is 405–406 MB. The shared region encoder is approximately 344 MB and is needed only once. To evaluate one model, download its checkpoint and the shared encoder; the other three model files are optional.

These weights are distributed as [GitHub release assets](https://github.com/Air000/CaB-ReID/releases/tag/checkpoints), separately from the source code. Cloning this repository alone does not download the large weight files. The checksum manifest identifies the evaluation files, and each checkpoint also records the SHA-256 of its original full checkpoint. They cannot be used to resume training.

```bash
python tools/verify_checkpoints.py --dataset mvti
# Or, for the two TRC-31K checkpoints:
python tools/verify_checkpoints.py --dataset trc31k
# If you downloaded only one backbone, also add --backbone clipreid or --backbone transreid.
```

Evaluation constructs the visual models locally and loads the supplied weights. No CLIP or ImageNet pretrained-weight download is needed, and no retraining is required. Keep `cabreid_region_encoder.pth` beside the selected checkpoint, even if you move the weights to another directory.

### 4. Evaluate

For MV-TI:

```bash
python main.py evaluate --backbone clipreid --dataset mvti \
  --data-root ../dataset/MV-TI \
  --weight weights/cabreid_clipreid_mvti.pth

python main.py evaluate --backbone transreid --dataset mvti \
  --data-root ../dataset/MV-TI \
  --weight weights/cabreid_transreid_mvti.pth
```

For TRC-31K, once the dataset is available:

```bash
python main.py evaluate --backbone clipreid --dataset trc31k \
  --data-root ../dataset/TRC-31K \
  --weight weights/cabreid_clipreid_trc31k.pth

python main.py evaluate --backbone transreid --dataset trc31k \
  --data-root ../dataset/TRC-31K \
  --weight weights/cabreid_transreid_trc31k.pth
```

The CLI selects the matching configuration: MV-TI uses `256 × 128` images and TRC-31K uses `256 × 256`, both with stride 12. Both backbones use binary masks at threshold 0.45, exclusive body pooling, Global/Cab/Body fusion weights `1.0/0.5/0.5`, no part-validity filtering, normalised features, and no reranking. TransReID retains its four native local branches.

Evaluation prints mAP and CMC Rank-1/5/10 and saves `test_log.txt` under `outputs/clipreid_mvti/`, `outputs/transreid_mvti/`, `outputs/clipreid/`, or `outputs/transreid/`. Gallery images sharing both identity and camera with the query are excluded.

For a smaller GPU, append `-- TEST.IMS_PER_BATCH 16`. To inspect a command without loading data or models, add `--dry-run`.

## Training

Training from pretrained backbones remains available through the same CLI; the smaller evaluation checkpoints are not training checkpoints. Training requires the corresponding dataset and follows the native schedule: 60 prompt-learning epochs plus 60 Re-ID epochs for CLIP-ReID, or 120 epochs for TransReID. CLIP ViT-B/16 is downloaded automatically for training.

For TransReID training only, download the ImageNet-pretrained ViT-B/16 weights:

```bash
wget -O weights/jx_vit_base_p16_224-80ecf9dd.pth \
  https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
```

```bash
python main.py train --backbone clipreid --dataset mvti --data-root ../dataset/MV-TI
python main.py train --backbone transreid --dataset mvti --data-root ../dataset/MV-TI
```

For TRC-31K, use `--dataset trc31k --data-root ../dataset/TRC-31K` after obtaining access. The paper training batch size is 64; changing it changes the training recipe.

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

After TRC-31K is released, researchers can use its query/gallery protocol to compare other truck Re-ID methods directly with CaB-ReID. Its CSV manifests are authoritative: each row records the relative image path, identity, and zero-based camera index. Please report the dataset, configuration, backbone, image size, checkpoint checksum, and whether reranking was used.

The `cabreid` package is separated from the backbone implementations so that its masking, pooling, fusion, and memory components can be integrated into other token-based Re-ID models.

## Privacy and responsible use

The release images were reviewed for visible registration plates and real-person faces. The public pixels contain strong blur over 109 manually verified regions in 97 images. Pre-anonymisation images, reviewer records, curation backups, and intermediate audit material are not distributed.

TRC-31K is intended for academic research on vehicle re-identification and representation learning. Users are responsible for complying with the final dataset licence, applicable privacy requirements, and institutional policies.

## Citation

The associated journal manuscript has not yet received a permanent bibliographic record. A verified BibTeX entry will be added after publication. Until then, please identify the resource as **CaB-ReID with TRC-31K** and link to this project page.

## Licence and access

The upstream CLIP-ReID and TransReID licences are retained inside their respective source directories. The CaB-ReID project licence and TRC-31K dataset licence are pending approval. Do not redistribute this release candidate until those terms are approved.
