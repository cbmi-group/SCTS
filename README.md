
# The official implementation of the paper : SCTS: Instance Segmentation of Single Cells Using a Transformer-Based Semantic-Aware Model and Space-Filling Augmentation

## dataset:

### HEK293T dataset [download](https://drive.google.com/file/d/1CH8MI_FXjQwN5VhhUQYNYfgS2ZRE1GMt/view?usp=share_link)
HEK293T is a small-scale in-house dataset of HEK293T cells imaged by confocal microscopy. It is partitioned into 108 training images and 37 test images, with a total of 2012 training instances and 576 test instances. It is characterized by uneven cell brightness and the presence of some weak signal regions.

### LIVECELL dataset [download](https://sartorius-research.github.io/LIVECell/)
LIVECell is a large-scale public dataset that consists of 5239 phase-contrast microscopy images with a total of 1,686,352 cell instances from eight different cell types. Images in LIVECell have two notable features. First, they show large variations in cell density, with images of cells growing from the initial seeding phase to a fully confluent monolayer . In the case of full confluency, a LIVECell image 704×520 in size can contain more than 3000 instances, which makes it difficult even for human eyes to accurately identify cell boundaries. Second, cells in the LIVECell dataset show a wide variety in size and shape, including small and round BV-2 cells, large and flat SK-OV-3 cells, and elongated SH-SY5Y cells. The high density and wide variety of cells pose substantial challenges for the design of instance segmentation algorithms.

## Code Usage

### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation and dataset preparation.

### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```

### Training

To train a detector with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```
For example, to train a SCTS model with a `Swin-T` backbone and 8 gpus, run:
```
tools/dist_train.sh configs/scts/scts.py 8 --cfg-options model.pretrained=<PRETRAIN_MODEL> 
```

**Note:** `use_checkpoint` is used to save GPU memory. Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.


### Apex (optional):
We use apex for mixed precision training by default. To install apex, run:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
If you would like to disable apex, modify the type of runner as `EpochBasedRunner` and comment out the following code block in the [configuration files](configs/swin):
```
# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
```
### Conference
IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2023
