# Illuminating Pedestrians via Simultaneous Detection & Segmentation

Garrick Brazil, Xi Yin, Xiaoming Liu

## Introduction

Pedestrian detection framework using simultaneous detection and segmentation as detailed in [arXiv report](https://arxiv.org/abs/1706.08564), accepted to ICCV 2017. 

Our SDS-RCNN framework is derivative work of [Faster R-CNN](https://github.com/ShaoqingRen/faster_rcnn) and [RPN+BF](https://github.com/zhangliliang/RPN_BF). Tested with Ubuntu 14.04, CUDA 7.5, Matlab 2016a, Titan X GPU, and a modified version of Caffe v1.0 as provided. Unless otherwise stated the below scripts and instructions assume cwd in MATLAB is the project root of SDS-RCNN. 


    @inproceedings{brazil2017illuminating,
        title={Illuminating Pedestrians via Simultaneous Detection \& Segmentation},
        author={Brazil, Garrick and Yin, Xi and Liu, Xiaoming},
        booktitle={Proceedings of the IEEE International Conference on Computer Vision},
        address={Vencie, Italy},
        year={2017}
    }

## Setup

- **Build Caffe**

    Build caffe and matcaffe following the usual [instructions](http://caffe.berkeleyvision.org/installation.html). We provide an upgraded version of Caffe v1.0 which includes the required layers necessary to run Faster R-CNN in *external/caffe*.

- **Data**

    Download the full [Caltech](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/) dataset. In order to evaluate you must extract or soft-link a folder called *data-USA* into the directory *external/caltech_toolbox/* such that such that the annotation and video files can be accessed as: *data-USA/annotations/\*.vbb* and *data-USA/videos/\*.seq*.

    Then extract the datasets for train, val, test in Matlab as below (or setup softlinks as desire). 

    ```
    dbInfo('usatrain');    dbExtract('datasets/caltechx10/train', 1, 3);
    dbInfo('usatrainval'); dbExtract('datasets/caltechval/val', 1);
    dbInfo('usatest');     dbExtract('datasets/caltechx1/test', 1);
    ```
- **Misc**
    1. Download the pretrained [VGG16](https://www.cse.msu.edu/computervision/vgg16.zip) on ImageNet and place in *SDS-RCNN/pretrained/vgg16.caffemodel*.
    1. Run *build_nms* to compile nms mex files.
    1. Review the config files in *experiments/+Config/+[rcnn|rpn]* for additional information.

## Training

Training both stages takes about 18 hours on a single Titan X.

``` matlab
rpn_config  = 'caltech_VGG16_weak_seg';
rcnn_config	= 'caltech_VGG16_weak_seg';
gpu_id = 1;

% train both stages
train_all(rpn_config, rcnn_config, gpu_id);
```

## Testing

We provide the collective SDS-RCNN trained models for RPN and BCN (7.36% MR), as well as the RPN only file with cost-sensitive off (9.63% MR). There are associated artifact files of anchors, bbox_stds, bbox_means, and basic configurations which should be loaded into memory at test time as depicted below. All files are packed into [SDS-RCNN-Release.zip](https://www.cse.msu.edu/computervision/SDS-RCNN-Release.zip).

``` matlab
load('rpn_conf.mat');
load('rcnn_conf.mat');
load('anchors.mat');
load('bbox_means.mat');
load('bbox_stds.mat');
gpu_id = 1;

% test RPN only
test_rpn(test_prototxt_path, weights_path, rpn_conf, anchors, bbox_means, bbox_stds, gpu_id)

% test RPN and BCN (full SDS-RCNN)
test_rcnn(test_prototxt_path, weights_path, rpn_conf, anchors, bbox_means, bbox_stds, ...
    rcnn_prototxt, rcnn_weights, rcnn_conf, gpu_id)

```
