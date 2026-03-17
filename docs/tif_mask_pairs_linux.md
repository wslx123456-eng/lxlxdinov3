# TIFF/PNG linear segmentation on Linux

This repository now includes a linear-probing style semantic segmentation path for private TIFF/PNG datasets.

## Task assumptions

- Input image: RGB `tif` or `png`
- Label image: single-channel class map
- Raw label values: `0`, `80`, `160`, `240`
- Train objective: validate DINOv3 feature quality with a linear segmentation head

The training path remaps labels as follows:

- `0 -> 0`
- `80 -> 1`
- `160 -> 2`
- `240 -> 3`

The segmentation head outputs `num_classes=4` channels and uses cross entropy loss.

## Dataset layout

Default layout:

```text
<dataset_root>/
  train/
    images/
      *.tif or *.png
    masks/
      *.tif or *.png
  val/
    images/
      *.tif or *.png
    masks/
      *.tif or *.png
```

If your secure machine stores data differently, override:

- `datasets.train_images`
- `datasets.train_masks`
- `datasets.val_images`
- `datasets.val_masks`

## Config to edit before training

Edit `dinov3/eval/segmentation/configs/config-tif-mask-pairs-linear-training.yaml`:

- `output_dir`
- `model.config_file`
- `model.pretrained_weights`
- `datasets.root`
- `bs`
- `scheduler.total_iter`
- `transforms.train.img_size`
- `transforms.eval.img_size`

For a ViT-B/16 backbone, the provided template already points to:

- `dinov3/configs/train/multidist_tests/vitb_p16.yaml`

The weights path still needs to match the actual location on the Linux node.

## Linux setup

Create the environment:

```bash
bash scripts/setup_tif_linear_linux.sh dinov3_tif https://download.pytorch.org/whl/cu124
```

The second argument is the PyTorch wheel index URL. Change it if the secure machine needs a different CUDA build.

## Launch training

Single-GPU launch:

```bash
bash scripts/train_tif_linear_linux.sh \
  dinov3/eval/segmentation/configs/config-tif-mask-pairs-linear-training.yaml
```

Equivalent direct command:

```bash
PYTHONPATH=. torchrun --nproc_per_node=1 dinov3/eval/segmentation/run.py \
  config=dinov3/eval/segmentation/configs/config-tif-mask-pairs-linear-training.yaml
```

## What was changed for this path

- Custom `tif_mask_pairs` dataset loader
- Label remapping for `0/80/160/240`
- Linear-only segmentation training path
- Photometric distortion disabled by default
- Final validation in the training loop now runs only once after the last step when needed
