# LapSRN-PyTorch

### Overview

This repository contains an op-for-op PyTorch reimplementation of [Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution](https://arxiv.org/pdf/1704.03915.pdf).

### Table of contents

- [LapSRN-PyTorch](#lapsrn-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [About Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution](#about-deep-laplacian-pyramid-networks-for-fast-and-accurate-super-resolution)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
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

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains DIV2K, DIV8K, Flickr2K, OST, T91, Set5, Set14, BSDS100 and BSDS200, etc.

- [Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot)

## Test

Modify the contents of the file as follows.

- line 30: `upscale_factor` change to the magnification you need to enlarge.
- line 32: `mode` change Set to valid mode.
- line 70: `model_path` change weight address after training.

## Train

Modify the contents of the file as follows.

- line 30: `upscale_factor` change to the magnification you need to enlarge.
- line 32: `mode` change Set to train mode.

If you want to load weights that you've trained before, modify the contents of the file as follows.

- line 47: `start_epoch` change number of training iterations in the previous round.
- line 48: `resume` change to `True`.

## Result

Source of original paper results: https://arxiv.org/pdf/1704.03915.pdf

In the following table, the value in `()` indicates the result of the project, and `-` indicates no test.

| Dataset | Scale |       PSNR       | 
|:-------:|:-----:|:----------------:|
|  Set5   |   2   | 37.25(**37.19**) |
|  Set5   |   4   | 31.33(**31.16**) |
|  Set5   |   8   | 26.14(**25.80**) |

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

[[Paper]](https://arxiv.org/pdf/1704.03915.pdf) [[Author's implements(MATLAB)]](https://github.com/phoenix104104/LapSRN) 

```
@inproceedings{LapSRN,
    author    = {Lai, Wei-Sheng and Huang, Jia-Bin and Ahuja, Narendra and Yang, Ming-Hsuan}, 
    title     = {Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution}, 
    booktitle = {IEEE Conferene on Computer Vision and Pattern Recognition},
    year      = {2017}
}
```
