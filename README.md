# DynamicPAE

## Overview

This project contains the official implementation for the IEEE TPAMI 2025 Paper [**DynamicPAE: DynamicPAE: Generating Scene-Aware Physical Adversarial Examples in Real-Time**](https://ieeexplore.ieee.org/document/11219170).

## Directory Structure

This repository follows the directory structure of [https://github.com/hujinCN/aiworkflow](https://github.com/hujinCN/aiworkflow).

model definition: **[models](sources_root/dynamic_example/models)** 

simulation stages definition: [stages.py](sources_root/tasks/scenarios.py)

base trainer module & digital classification module: [bask_atk_model.py](sources_root/trainer/base_atk_model.py)

## Quick Start Guide

### Prerequisites


1. install torch & torchvision. verified on torch 2.0.1 and 2.7.0. 

Please refer to the official installation code on pytorch.org.

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

2. install other packages. Note that ultralytics version needs to be recheck, since this package is under development
```shell
conda install pytorch-lightning==2.2.1 seaborn loguru tensorboard --yes
pip install ultralytics==8.3.29 easydict pycocotools timm einops pytorch-msssim lpips torchmetrics==1.2.0
pip install diffusers torchattacks kornia imgaug openai-clip opencv-python matplotlib
```

3. prepare detlib

See [README.md](sources_root/det_root/detlib/README.md), section _Pretrained models_.

This submodule is from [T-SEA](https://github.com/VDIGPKU/T-SEA) project. See [LICENSE](https://github.com/VDIGPKU/T-SEA?tab=readme-ov-file#license).

4. Download the datasets.
   
Put all datasets under a directory, and modify the path in [path_cfg.yml](workflow/path_cfg.yml).

For INRIA dataset, please convert to COCO format.

The prepared dataset directory structure shall be as follows:
```
coco
|-- train2017
|-- val2017
|-- annotations
INRIAPerson
|-- Train
|-- Test
|-- test_ann_xywh.json
|-- train_ann_xywh.json
```

### Running the Experiment

1. Model Training:
   ```bash
   cd sources_root/dynamic_example
   bash run.sh
   ```
   Approximate time: 1 day for single 4090 GPU. Result may be different due to randomness.

2. Model Evaluation:
   ```bash
   cd sources_root/dynamic_example
   bash test.sh
   ```

Refer to the shell scripts for more details.


### Interpreting Results
The results are stored in the logs directory. Please refer to [https://github.com/hujinCN/aiworkflow](https://github.com/hujinCN/aiworkflow) for format details.

# Citation
```
@ARTICLE{hu2025dynamicpae,
  author={Hu, Jin and Liu, Xianglong and Wang, Jiakai and Zhang, Junkai and Yang, Xianqi and Qin, Haotong and Ma, Yuqing and Xu, Ke},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={DynamicPAE: Generating Scene-Aware Physical Adversarial Examples in Real-Time}, 
  year={2026},
  volume={48},
  number={3},
  pages={2413-2430},
  doi={10.1109/TPAMI.2025.3626068}
}
```

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
