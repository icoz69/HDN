# Hierarchical Normalization for Robust Monocular Depth Estimation

This repo contains PyTorch re-implementation of HDN loss functions, proposed in  

[Hierarchical Normalization for Robust Monocular Depth Estimation](https://openreview.net/pdf?id=BNqRpzwyOFU)" (NeurIPS 2022)  



If you use the code in this repo for your work, please cite the following bib entries:


    @article{zhang2022hierarchical,
      title={Hierarchical Normalization for Robust Monocular Depth Estimation},
      author={Zhang, Chi and Yin, Wei and Wang, Zhibin and Yu, Gang and Fu, Bin and Shen, Chunhua},
      journal= {NeurIPS},
      year={2022}
    }


If you have any questiones regarding the paper, please send a email to the author `johnczhang[at]tencent[dot]com`.

## Abstract

In this paper, we address monocular depth estimation with deep neural networks. To enable training of deep monocular estimation models with various sources of datasets, state-of-the-art methods adopt image-level normalization strategies to generate affine-invariant depth representations. However, learning with image-level normalization mainly emphasizes the relations of pixel representations with the global statistic in the images, such as the structure of the scene, while the fine-grained depth difference may be overlooked. In this paper, we propose a novel multi-scale depth normalization method that hierarchically normalizes the depth representations based on spatial information and depth distributions. Compared with previous normalization strategies applied only at the holistic image level, the proposed hierarchical normalization can effectively preserve the fine-grained details and improve accuracy. We present two strategies that define the hierarchical normalization contexts in the depth domain and the spatial domain, respectively. Our extensive experiments show that the proposed normalization strategy remarkably outperforms previous normalization methods, and we set new state-of-the-art on five zero-shot transfer benchmark datasets.

## Usage
Refer to `demo.py`.

## Acknowledgment
This project references the codes in the following repos.
- [Omnidata](https://github.com/EPFL-VILAB/omnidata)


