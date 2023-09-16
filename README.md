# MILA: Memory-Based Instance-Level Adaptation for Cross-Domain Object Detection
by Onkar Krishna, Hiroki Ohashi and Saptarshi Sinha.

This repository contains the code for the paper '[MILA: Memory-Based Instance-Level Adaptation for Cross-Domain Object Detection](https://arxiv.org/abs/2309.01086),' which has been accepted for oral presentation at BMVC 2023.

## Requirements
The environment required to successfully reproduce our results primarily includes.
```
Python >= 3.8
CUDA == 10.1
PyTorch == 1.7.0+cu101
detectron2 == 0.5 
```
Please refer to the [instructions](https://github.com/facebookresearch/detectron2/releases) for guidance on installing Detectron2

## Datasets 

Please download and arrange the following datasets:
- [Cityscapes](https://www.cityscapes-dataset.com/downloads/)
- [PASCAL VOC 07+12](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
- [Clipart, Watercolor](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets) 

Ensure that you organize these datasets in the same manner as demonstrated in the [Adaptive Teacher](https://github.com/facebookresearch/adaptive_teacher) repository.

Additionally, for the following datasets:
- [Sim10k](https://fcav.engin.umich.edu/projects/driving-in-the-matrix)
- [Comic2k](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets)

Please arrange them as following:
```shell
MILA/
└── datasets/
    └── sim10k/
        ├── Annotations/
        ├── ImageSets/
        └── JPEGImages/
   └── comic/
        ├── Annotations/
        ├── ImageSets/
        └── JPEGImages/
```
## How to run the code

- Train the MILA using Sim10k as the source domain and Cityscapes as the target domain
  
```shell
python train_net_mem.py \
      --num-gpus 4 \
      --config configs/faster_rcnn_R101_cross_sim10k_13031.yaml\
      OUTPUT_DIR output/sim10k_ckpt
```

- Train the MILA with Pascal VOC as the source domain and Comic2k as the target domain.

```shell
python train_net_mem.py \
      --num-gpus 1 \
      --config configs/faster_rcnn_R101_cross_comic_08032.yaml\
      OUTPUT_DIR output/comic_ckpt
```

- Train the MILA with Cityscapes as the source domain and Foggy Cityscapes as the target domain. you need to install the following packages: pip install cityscapesScripts and pip install shapely.

```shell
python train_net_mem.py \
      --num-gpus 1 \
      --config configs/faster_rcnn_VGG_cross_city_07021_3.yaml \
      OUTPUT_DIR output/foggy_ckpt
```
- For evaluation,

```shell
python train_net_mem.py \
      --eval-only \
      --num-gpus 4 \
      --config configs/faster_rcnn_R101_cross_sim10k_13031.yaml \
      MODEL.WEIGHTS <your weight>.pth
```
## Acknowledgement
This repository is based on the code from [Adaptive Teacher](https://github.com/facebookresearch/adaptive_teacher). We thank authors for their contributions.

## Citation

If you use our code, please consider citing our paper as
```BibTeX
@article{krishna2023mila,
  title={MILA: Memory-Based Instance-Level Adaptation for Cross-Domain Object Detection},
  author={Krishna, Onkar and Ohashi, Hiroki and Sinha, Saptarshi},
  journal={arXiv preprint arXiv:2309.01086},
  year={2023}
}
```

For queries, contact at onkar.krishna.vb@hitachi.com


