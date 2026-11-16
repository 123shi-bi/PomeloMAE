# PomeloMAE

PomeloMAE pre-trains a ViT-B/16 using four RGB views of each pomelo. The visible patches from the four views are jointly encoded, and a shared decoder reconstructs each view. The pre-trained encoder is then fine-tuned to predict soluble solids content (SSC, in °Brix) from a single image.

## Setup

Run the commands from this directory in a Python 3.12 environment. The code was tested with PyTorch 2.5.1 and torchvision 0.20.1. To install the CUDA 11.8 builds used for training, run the first command before installing the remaining dependencies.

```sh
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

The commands below default to `--device cuda`. Use `--device cpu` for CPU execution.

## Data layout

Use a fruit-level split. Place all four views of a fruit in the same directory, with no fruit ID shared across `train`, `val`, and `test`.

```text
data/
├── train/
│   └── class_name/
│       └── 1001/
│           ├── 1001_a.jpg
│           ├── 1001_b.jpg
│           ├── 1001_c.jpg
│           └── 1001_d.jpg
├── val/                         # same class/fruit/file layout
└── test/                        # same class/fruit/file layout
```

The view suffixes `a`, `b`, `c`, and `d` denote front, right, back, and left. Use numeric fruit IDs and one SSC measurement per fruit in a CSV. The `fruit_id` values must match the fruit directory names exactly. Pre-training reads only `train` and needs no measurements.

```csv
fruit_id,ssc
1001,12.4
```

Supply your own images and measurements. The repository contains source code and configuration for its dependencies.

## Pre-training, fine-tuning, and evaluation

```sh
python -m pomelomae.main_pretrain_cmr --data-path data --output-dir runs/seed0/pretrain --seed 0
python -m pomelomae.finetune_ssc_regression --data-path data --measurements measurements.csv --finetune runs/seed0/pretrain/checkpoint-99.pth --output-dir runs/seed0/finetune --seed 0
python -m pomelomae.evaluate_ssc_checkpoint --data-path data --measurements measurements.csv --checkpoint runs/seed0/finetune/checkpoint-best-image-r2.pth --output-dir runs/seed0/test --seed 0
```

To run all three seeds, repeat the commands with `--seed 1` and `--seed 2` and use a separate output directory for each seed. Fine-tuning selects the checkpoint with the highest image-level validation R². Evaluation writes separate image-level and fruit-level metrics and predictions. The fruit-level prediction is the mean of the four image predictions.

## Default configuration

| Parameter | Pre-training | Fine-tuning |
|---|---|---|
| Epochs | 100 | 50 |
| Batch size | 16 fruits × 4 views = 64 images | 64 individual images |
| Peak learning rate | 1.5e-4 | 5e-4 |
| Warm-up epochs | 10 | 5 |
| Minimum learning rate | 0 | 1e-6 |
| Weight decay | 0.05 | 0.05 |
| Input | 224×224 RGB, 16×16 patches | 224×224 RGB, 16×16 patches |
| Mask ratio per view | 0.80 | — |
| Target-quadrant share of visible patches | 0.70 | — |
| Objective | Masked-patch MSE with patch-normalized pixel targets | Huber (δ = 1.0) + 0.2 × (1 − CCC), on standardized SSC |

The single-GPU pre-training recipe uses one batch per optimizer step. Each view keeps 39 of 196 patches. The target quadrant receives 27 priority slots, with the remaining slots filled by random sampling outside that quadrant when available. The foreground rule is applied independently to each augmented view. RGB values are in [0, 1]. Candidate pixels satisfy (R + G)/2 − B > 0.12, max(R, G, B) − min(R, G, B) > 0.12, and (R + G + B)/3 > 0.20. A patch is eligible as a foreground patch when at least 25% of its pixels pass the rule.

The shared Transformer decoder has 8 blocks, a hidden size of 512, 16 attention heads per block, and an MLP width of 2048. The fine-tuning model takes the normalized `[CLS]` token from the final ViT block and feeds it to a three-layer regression head with widths 768 → 256 → 64 → 1, LayerNorm at the input, and GELU and dropout (0.1) after the two hidden layers. Fine-tuning uses layer-wise learning-rate decay of 0.65 and a maximum drop-path rate of 0.1.

## License and Attribution

The PomeloMAE authors developed the four-view joint encoding and reconstruction pipeline, fruit-aware complementary masking, and single-image SSC regression. These original contributions are released under the [Creative Commons Attribution–NonCommercial 4.0 International License](LICENSE).

Portions of the ViT model and training utilities are adapted from Meta Platforms, Inc. and affiliates’ Masked Autoencoders (MAE) implementation. The relevant source files retain their original copyright and license notices. [Upstream MAE source](https://github.com/facebookresearch/mae)
