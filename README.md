# LapSRN-PyTorch

### Overview

This repository contains an op-for-op PyTorch reimplementation
of [Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution](https://arxiv.org/pdf/1704.03915.pdf).

### Table of contents

- [LapSRN-PyTorch](#lapsrn-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [About Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution](#about-deep-laplacian-pyramid-networks-for-fast-and-accurate-super-resolution)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
        - [Download train dataset](#download-train-dataset)
        - [Download valid dataset](#download-valid-dataset)
    - [Test](#test)
    - [Train](#train)
    - [Result](#result)
    - [Credit](#credit)
        - [Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution](#deep-laplacian-pyramid-networks-for-fast-and-accurate-super-resolution)

## About Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution

If you're new to LapSRN, here's an abstract straight from the paper:

Convolutional neural networks have recently demonstrated high-quality reconstruction for single-image superresolution. In this paper, we propose the
Laplacian Pyramid Super-Resolution Network (LapSRN) to progressively reconstruct the sub-band residuals of high-resolution images. At each pyramid
level, our model takes coarse-resolution feature maps as input, predicts the high-frequency residuals, and uses transposed convolutions for upsampling
to the finer level. Our method does not require the bicubic interpolation as the pre-processing step and thus dramatically reduces the computational
complexity. We train the proposed LapSRN with deep supervision using a robust Charbonnier loss function and achieve high-quality reconstruction.
Furthermore, our network generates multi-scale predictions in one feed-forward pass through the progressive reconstruction, thereby facilitates
resource-aware applications. Extensive quantitative and qualitative evaluations on benchmark datasets show that the proposed algorithm performs
favorably against the state-of-the-art methods in terms of speed and accuracy.

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/112QV7vQwFNHEC7DDZoJWAmW8tNOgHmE0?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1MLhm_TZL5gNWLOOKxmm9JA) access:`llot`

## Download datasets

### Download train dataset

#### TB291

- Image format
    - [Google Driver](https://drive.google.com/drive/folders/13wiE6YqIhyix0RFxpFONJ7Zz_00CttdX?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1mhbFj0Nvwthmgx07Gas5BQ) access: `llot`

- LMDB format (train)
    - [Google Driver](https://drive.google.com/drive/folders/1BPqN08QHk_xFnMJWMS8grfh_vesVs8Jf?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1eqeORnKcTmGatx2kAG92-A) access: `llot`

- LMDB format (valid)
    - [Google Driver](https://drive.google.com/drive/folders/1bYqqKk6NJ9wUfxTH2t_LbdMTB04OUicc?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1W34MeEtLY0m-bOrnaveVmw) access: `llot`

### Download valid dataset

#### Set5

- Image format
    - [Google Driver](https://drive.google.com/file/d/1GtQuoEN78q3AIP8vkh-17X90thYp_FfU/view?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1dlPcpwRPUBOnxlfW5--S5g) access:`llot`

#### Set14

- Image format
    - [Google Driver](https://drive.google.com/file/d/1CzwwAtLSW9sog3acXj8s7Hg3S7kr2HiZ/view?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1KBS38UAjM7bJ_e6a54eHaA) access:`llot`

#### BSD200

- Image format
    - [Google Driver](https://drive.google.com/file/d/1cdMYTPr77RdOgyAvJPMQqaJHWrD5ma5n/view?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1xahPw4dNNc3XspMMOuw1Bw) access:`llot`

## Test

Modify the contents of the file as follows.

- line 24: `upscale_factor` change to the magnification you need to enlarge.
- line 25: `mode` change Set to valid mode.
- line 91: `model_path` change weight address after training.

## Train

Modify the contents of the file as follows.

- line 24: `upscale_factor` change to the magnification you need to enlarge.
- line 25: `mode` change Set to train mode.

If you want to load weights that you've trained before, modify the contents of the file as follows.

- line 53: `resume` change to `True`.
- line 54: `strict` Transfer learning is set to `False`, incremental learning is set to `True`.
- line 55: `start_epoch` change number of training iterations in the previous round.
- line 56: `resume_weight` the weight address that needs to be loaded.

## Result

Source of original paper results: https://arxiv.org/pdf/1704.03915.pdf

In the following table, the value in `()` indicates the result of the project, and `-` indicates no test.

| Dataset | Scale |       PSNR       | 
|:-------:|:-----:|:----------------:|
|  Set5   |   2   | 37.25(**37.19**) |
|  Set5   |   4   | 31.33(**31.20**) |
|  Set5   |   8   |   26.14(**-**)   |

Low Resolution / Super Resolution / High Resolution
<span align="center"><img src="assets/result.png"/></span>

### Credit

#### Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution

_Lai, Wei-Sheng and Huang, Jia-Bin and Ahuja, Narendra and Yang, Ming-Hsuan_ <br>

**Abstract** <br>
Convolutional neural networks have recently demonstrated high-quality reconstruction for single-image superresolution. In this paper, we propose the
Laplacian Pyramid Super-Resolution Network (LapSRN) to progressively reconstruct the sub-band residuals of high-resolution images. At each pyramid
level, our model takes coarse-resolution feature maps as input, predicts the high-frequency residuals, and uses transposed convolutions for upsampling
to the finer level. Our method does not require the bicubic interpolation as the pre-processing step and thus dramatically reduces the computational
complexity. We train the proposed LapSRN with deep supervision using a robust Charbonnier loss function and achieve high-quality reconstruction.
Furthermore, our network generates multi-scale predictions in one feed-forward pass through the progressive reconstruction, thereby facilitates
resource-aware applications. Extensive quantitative and qualitative evaluations on benchmark datasets show that the proposed algorithm performs
favorably against the state-of-the-art methods in terms of speed and accuracy.

[[Code]](http://vllab.ucmerced.edu/wlai24/LapSRN) [[Paper]](https://arxiv.org/pdf/1704.03915.pdf)

```
@inproceedings{LapSRN,
    author    = {Lai, Wei-Sheng and Huang, Jia-Bin and Ahuja, Narendra and Yang, Ming-Hsuan}, 
    title     = {Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution}, 
    booktitle = {IEEE Conferene on Computer Vision and Pattern Recognition},
    year      = {2017}
}
```
