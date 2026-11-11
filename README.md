# Cycle-GAN-re-implementation-Project

An independent, from-scratch PyTorch re-implementation of **CycleGAN** ([Zhu et al., ICCV 2017](https://arxiv.org/abs/1703.10593)) for unpaired image-to-image translation (horse ↔ zebra), written without reference to the authors' official code. It features a 9-block ResNet generator, 70×70 PatchGAN discriminators, LSGAN loss, and an image history pool, following the paper's architecture and training recipe. The project also includes an **ablation study** on the identity loss and the cycle-consistency weight, evaluated qualitatively and with FID on the `horse2zebra` dataset.




---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Testing](#testing)
- [Evaluation (FID)](#evaluation-fid)
- [Implementation Details](#implementation-details)
- [Ablation Study and Results](#ablation-study-and-results)
- [Limitations](#limitations)
- [Reference](#reference)

---

## Overview

Classic image-to-image translation methods such as pix2pix need *paired* training data, which is expensive or impossible to collect in many settings (there is no photo of the same scene with both a horse and a zebra in the same pose). CycleGAN learns a mapping between two image domains **A** and **B** using only two unrelated image collections.

It learns two generators and two discriminators:

| Network | Role |
|---|---|
| `G : A → B` | translates domain A (e.g. horse) into domain B (e.g. zebra) |
| `F : B → A` | translates domain B back into domain A |
| `D_B` | distinguishes real B images from `G(A)` |
| `D_A` | distinguishes real A images from `F(B)` |

Cycle consistency (`F(G(a)) ≈ a` and `G(F(b)) ≈ b`) prevents mode collapse and keeps the structure of the input image.

## Features

- 9-block ResNet generator (6 blocks recommended for 128×128 images), following Johnson et al.
- 70×70 PatchGAN discriminators
- LSGAN (least-squares) adversarial objective
- L1 cycle-consistency loss and optional identity loss
- Image history buffer (pool of 50) to stabilise discriminator training
- Linear learning-rate decay schedule
- **Pause and resume** support: `latest.pth` is saved after every completed epoch and restores weights, optimizer state, and LR schedule position
- Unpaired data loader with random sampling from the second domain
- FID evaluation script based on InceptionV3 features
- Modular code: six small files, easy to read and modify

## Project Structure

```
.
├── datasets.py       # UnalignedDataset: unpaired loading + augmentation
├── models.py         # ResNet generator, PatchGAN discriminator, GAN loss, init
├── utils.py          # ImagePool, LR lambda, tensor/image conversion helpers
├── train.py          # Training loop, checkpointing, resume
├── test.py           # Inference: translate test images in both directions
├── metrics.py        # FID computation
└── requirements.txt
```

Expected dataset layout:

```
datasets/horse2zebra/
├── trainA/   # horses (training)
├── trainB/   # zebras (training)
├── testA/    # horses (testing)
└── testB/    # zebras (testing)
```

## Installation

```bash
git clone https://github.com/alirezaalimirzaei2003/Cycle-GAN-re-implementation-Project.git
cd Cycle-GAN-re-implementation-Project

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Requirements: `torch>=2.0`, `torchvision>=0.15`, `Pillow`, `numpy`, `scipy`.

A CUDA-capable GPU is strongly recommended. Pass `--gpu` to the scripts to use it.

## Dataset

The experiments use **horse2zebra** (1067 horse images and 1334 zebra images in the training set, 256×256). It can be downloaded from the original CycleGAN project page:

```bash
mkdir -p datasets && cd datasets
wget https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip
unzip horse2zebra.zip && cd ..
```

Any other unpaired dataset works as long as it follows the `trainA / trainB / testA / testB` layout.

## Training

Basic run (paper settings: 100 epochs constant LR + 100 epochs linear decay):

```bash
python train.py --dataroot ./datasets/horse2zebra --name horse2zebra --gpu
```

Shorter schedule (as used for the experiments in this project: 30 epochs total, with the LR decay starting at the halfway point):

```bash
python train.py --dataroot ./datasets/horse2zebra --name horse2zebra \
    --n_epochs 15 --n_epochs_decay 15 --gpu
```

Pause and resume. Press `Ctrl+C` at any time; re-run the same command with `--resume` to continue exactly where the last completed epoch ended:

```bash
python train.py --dataroot ./datasets/horse2zebra --name horse2zebra \
    --n_epochs 15 --n_epochs_decay 15 --gpu --resume
```

### Main arguments

| Argument | Default | Description |
|---|---|---|
| `--dataroot` | required | Folder containing `trainA/` and `trainB/` |
| `--name` | `cyclegan_run` | Experiment name (used for the checkpoint folder) |
| `--checkpoints_dir` | `./checkpoints` | Where checkpoints and samples are stored |
| `--load_size` / `--crop_size` | 286 / 256 | Resize size and random crop size |
| `--batch_size` | 1 | Batch size |
| `--ngf` / `--ndf` | 64 / 64 | Base filters of generator / discriminator |
| `--n_blocks` | 9 | ResNet blocks (9 for 256×256, 6 for 128×128) |
| `--n_epochs` | 100 | Epochs at constant learning rate |
| `--n_epochs_decay` | 100 | Epochs over which the LR decays linearly to 0 |
| `--lr` | 0.0002 | Adam learning rate |
| `--beta1` | 0.5 | Adam β1 |
| `--lambda_cyc` | 10.0 | Weight of the cycle-consistency loss |
| `--lambda_identity` | 0.5 | Identity loss weight relative to `lambda_cyc` (0 disables it) |
| `--pool_size` | 50 | Size of the generated-image history buffer |
| `--save_epoch_freq` | 5 | Save a named `epoch_N.pth` every N epochs |
| `--resume` | off | Resume from `checkpoints/<name>/latest.pth` |
| `--gpu` | off | Use CUDA if available |

### Outputs

```
checkpoints/<name>/
├── latest.pth        # full state (models + optimizers + epoch), updated every epoch
├── epoch_5.pth ...   # named checkpoints (models only)
└── samples/          # epochN_realA / fakeB / realB / fakeA preview images
```

## Testing

Translate test images in both directions (A→B and B→A):

```bash
python test.py \
    --dataroot ./datasets/horse2zebra \
    --checkpoint ./checkpoints/horse2zebra/latest.pth \
    --results_dir ./results/horse2zebra \
    --gpu
```

Results are saved to:

```
results/horse2zebra/
├── fakeB/   # A → B  (horse → zebra)
└── fakeA/   # B → A  (zebra → horse)
```

Use `--num_test` to control how many test images are translated (default 100). Make sure `--n_blocks` and `--ngf` match the values used during training.

## Evaluation (FID)

Compute the Fréchet Inception Distance between real target-domain images and generated images:

```bash
python metrics.py \
    --real_dir ./datasets/horse2zebra/testB \
    --fake_dir ./results/horse2zebra/fakeB \
    --gpu
```

Lower is better. FID compares the *statistics of two image sets* using InceptionV3 features. It does **not** check whether an individual output preserves the structure of its input (see [Ablation Study](#ablation-study-and-results)).

## Implementation Details

**Generator (`ResnetGenerator`)**: `c7s1-64 → d128 → d256 → 9 × R256 → u128 → u64 → c7s1-3 + Tanh`, with reflection padding, instance normalization and ReLU activations.

**Discriminator (`NLayerDiscriminator`)**: 70×70 PatchGAN. Four stride-2 convolutions followed by one stride-1 convolution and a 1-channel output map. LeakyReLU(0.2), instance normalization everywhere except the first layer. The output is a grid of real/fake scores, one per image patch.

**Objective** (generator side):

```
L = L_GAN(G, D_B) + L_GAN(F, D_A) + λ_cyc · L_cyc(G, F) + λ_cyc · λ_idt · L_idt(G, F)
```

- `L_GAN`: least-squares (MSE) loss instead of the log-likelihood loss
- `L_cyc`: L1 between `F(G(a))` and `a`, and between `G(F(b))` and `b`
- `L_idt`: L1 between `G(b)` and `b`, and between `F(a)` and `a`
- Defaults: `λ_cyc = 10`, `λ_idt = 0.5` (effective identity weight 5)

**Training details**
- Adam, `lr = 2e-4`, `β = (0.5, 0.999)`, batch size 1
- Constant LR for the first half of the schedule, then linear decay to zero
- Weights initialised from `N(0, 0.02)`
- Image pool of 50 for each discriminator (50 % chance to use a historical fake)
- Data augmentation: resize to 286, random crop to 256, random horizontal flip, normalisation to `[-1, 1]`
- Unpaired sampling: domain A is traversed in order, domain B is sampled randomly each step

## Ablation Study and Results

Three configurations were trained on horse2zebra for **30 epochs each** (instead of the 200 in the paper), on a single GTX 1070 Ti, about 30 hours in total:

| Configuration | Change | Command flags |
|---|---|---|
| **Base** | Paper settings | *(defaults)* |
| **no_identity** | Identity loss removed | `--lambda_identity 0` |
| **low_cyc** | Cycle weight reduced from 10 to 2 | `--lambda_cyc 2` |

> Note: the identity loss is scaled by `lambda_cyc`, so lowering `--lambda_cyc` also lowers the effective identity weight unless `--lambda_identity` is raised to compensate.

### Quantitative (FID on horse → zebra, lower is better)

| Model | FID |
|---|---|
| Base | 92.305 |
| no_identity | 119.847 |
| low_cyc | **76.700** |

### Qualitative observations

- **Base**: backgrounds, sky and curb colours are well preserved. Horses are not fully converted yet, and a person next to the horses is partly striped (typical of short training).
- **no_identity**: without the identity loss, the generator freely changes background colour, lighting and tone; the output drifts far from the input.
- **low_cyc**: with weaker cycle pressure, structure and fine detail are lost (for example a trash can turning into a zebra).

### Key takeaway

The qualitative trends match the paper: the identity loss preserves colour and tone, and the cycle-consistency loss preserves structure. However, `low_cyc`, which looks worst visually, obtains the **best FID**. This shows that FID only measures how closely the output distribution resembles real zebra images; it is blind to whether each output stays faithful to its input. This is consistent with the original authors' reliance on human evaluation and task-based metrics rather than FID.

## Limitations

- Models were trained for 30 epochs only, so absolute quality is lower than in the paper (200 epochs).
- FID is computed on a small number of images, with torchvision's ImageNet InceptionV3 (2048-d features), which differs from the standard TTUR/pytorch-fid implementation. Values are useful for comparing the runs here, not for comparing against published numbers.
- Each configuration was trained once with a single seed, so differences should be read as indicative rather than statistically conclusive.
- As in the paper, CycleGAN handles colour and texture changes well but struggles with large geometric changes, and it can inherit dataset biases.

## Reference

```bibtex
@inproceedings{zhu2017unpaired,
  title     = {Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author    = {Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A.},
  booktitle = {IEEE International Conference on Computer Vision (ICCV)},
  year      = {2017}
}
```


