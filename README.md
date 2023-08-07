# SFRNet
This is the official implement of SFRNet.



# SFRNet: Fine-Grained Oriented Object Recognition via Separate Feature Refinement

## Introduction
Fine-grained oriented object recognition is a practical need for intellectually interpreting remote sensing images. It aims at realizing fine-grained classification and precise localization with oriented bounding boxes, simultaneously. Our considerations for the task are general but decisive: 1) the extraction of subtle differences carries a big weight in differentiating fine-grained classes and 2) oriented localization prefers rotation-sensitive features. In this article, we propose a network with separate feature refinement (SFRNet), in which two transformer-based branches are designed to perform function-specific feature refinement for fine-grained classification and oriented localization, separately.

## Acknowledgement
Our SFRNet is implemented based on the [MMdetection](https://github.com/open-mmlab/mmdetection)

## Installation

### Requirements

- Linux or macOS (Windows is not currently officially supported)
- Python 3.6+
- PyTorch 1.6+
- CUDA 10.1+
- GCC 5+
- [mmcv 0.6.2](https://github.com/open-mmlab/mmcv)

### Install environment
a. Create a conda virtual environment and activate it.

```shell
conda create -n sfrnet python=3.7 -y
conda activate sfrnet
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

Note: Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

We install the mmdetction with CUDA 10.1 and pytorch 1.6.0. We recommand you using the same vision.

### Install BboxToolkit
```shell
cd BboxToolkit
pip install -v -e .  # or "python setup.py develop"
```

### Install mmdetection
a. Install mmcv
```shell
pip install mmcv==0.6.2
```

b. Install build requirements and then install mmdetection.
(We install our forked version of pycocotools via the github repo instead of pypi
for better compatibility with our repo.)

```shell
# back to mmdetection dir
pip install -r requirements/build.txt
pip install mmpycocotools
pip install pillow==6.2.2
pip install -v -e .  # or "python setup.py develop"
```

If you build mmdetection on macOS, replace the last command with

```
CC=clang CXX=clang++ CFLAGS='-stdlib=libc++' pip install -e .
```


## Usage
The expriments are conducted on FAIR1M datasets. To make data processing more convenient, we can convert the original datasets to the commonly used DOTA format. As a result, we take the DOTA dataset for example to introduce the training and testing procedure.

Here we show an example of recognition results on FAIR1M datasets.

![demo image](demo/recognition_results.jpg)

### Splitting images (for DOTA)
The DOTA images are too big to train. The characteristics of other datasets can also be compared to DOTA. We need to split the image before training.
```shell
cd BboxToolkit/tools
# Change the path of split_configs/xxxx.json
# add img_dir, ann_dir, and save_dir in xxx.json
python img_split.py --base_json split_configs/xxxx.json
```
The structure of splitted dataset is:
```
save_dir
├── images
│   ├──0001_0001.png
│   ├──0001_0002.png
│   ...
│   └──xxxx_xxxx.png
│
└── annfiles
    ├── split_config.json
    ├── patch_annfile.pkl
    └── ori_annfile.pkl

```
Where, we can reimplement the same splitting by `split_config.json`, the `patch_annfile.pkl` is the annotations after splitting, and `'ori_annfile.pkl` is the annotations before splitting.

Only need to add save_dir path in `configs/_base_/datasets/dota_*.py` to train and test the model.

## Testing

**Start testing**
```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [optional arguments]
```


## Training

**\*Important\***: The default learning rate in config files is for 1 GPUs and 2 img/gpu (batch size = 1*2 = 2).
According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you need to set the learning rate proportional to the batch size if you use different GPUs or images per GPU, e.g., lr=0.01 for 2 GPUs * 2 img/gpu and lr=0.02 for 4 GPUs * 2 img/gpu.

**Change dataset path**
```shell
cd configs/_base_/datasets/
# Change the path of the dota_*.py
# The path of dota_*.py is the dataset after splitting
```

**Start training**

```shell
# single-gpu training
python tools/train.py ${CONFIG_FILE} [optional arguments]

# multi-gpu training
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

**NOTE**
Thank you for your attention! The relevant codes will be constantly sorted out, uploaded and upgraded.


## License

This project is released under the [Apache 2.0 license](LICENSE).

