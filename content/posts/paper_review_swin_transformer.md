---
title: "Paper Review: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
date: "2021-03-27T23:46:37.121Z"
template: "post"
draft: false
slug: "paper-review-swin-transformer"
category: "Paper Review"
tags:
  - "Computer Vision"
description: "My review of the paper Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
socialImage: "/media/image-2.jpg"
---

[Paper link](https://arxiv.org/abs/2103.14030)

Code available [here](https://github.com/microsoft/Swin-Transformer) (no implementation at the moment of writing this review)

![collage](/media/paper_review_swin_transformer/review_collage.png)

An amazing paper from Microsoft Research Asia presents a brand new vision Transformer called Swin Transformer that 
can serve as a backbone just like usual CNNs in computer vision and Transformers in natural language processing (NLP).
<br><br>
There are two main problems with the usage of Transformers for computer vision. Firstly, existing Transformer-based models have tokens of a fixed scale. 
However in contrast to the word tokens, visual elements can be different in scale.
Secondly, computational complexity of self-attention is quadratic to image size, causing problems in vision tasks with dense
predictions at the pixel level.
<br><br>
Authors offer stratagies to solve these challenges:
* Hierarchical feature maps for convenient utilization of techniques like feature pyramid networks (FPN) or U-Net for dense predictions.
* Computing self-attention locally within non-overlapping windows with equal number of pathes to achieve linear complexity.

![fig 1](/media/paper_review_swin_transformer/fig_1.jpg)

Swin Transformer outperformes current state-of-the-art approaches on both COCO object detection and ADE20K semantic segmentation 
while achieving the best speed-accuracy trade-off on image classification.

## Methods

![fig 3](/media/paper_review_swin_transformer/fig_3.jpg)

#### Overall Architecture

1. Splitting RGB image into non-overlapping pathes (tokens).
2. Applying linear embedding layer to translate raw feature into an arbitrary dimension.
3. Applying several Swin Transformer blocks with modified self-attention computation and maintaining the number of tokens.
4. Reducing number of tokens by patch merging layers creating the same feature map resolutions like those in common CNNs.

#### Shifted Window based Self-Attention

Standard global self-attention is not quite suitable for representations of high-resolution images because of quadratic complexity.
Authors propose to compute self-attention within local windows.<br>
Here is the comparison of computational complexities of a global MSE module and a new window-based one
1. M is the size of *M x M* patch
2. Images consists of *h x w* patches

![complexities](/media/paper_review_swin_transformer/complexities.png)

Moreover, there is an idea about transferring shifted window partioning strategy to the next MSA block to create
additional connections across windows.<br>
Computation of Swin Transformer blocks:

![swin_blocks](/media/paper_review_swin_transformer/swin_blocks.png)

Shifted window partitioning will result in more windows and some of these windows will be smaller than *M x M*.
So, authors propose efficient batch computation approach with cyclic shifting toward the top-left direction.
Since batched window might be created of sub-windows which are not adjacent in the feature map, mask is applied.

![batched](/media/paper_review_swin_transformer/batched.png)

Also, relative position bias is used

![bias](/media/paper_review_swin_transformer/bias.png)

## Experiments

Experiments were made on ImageNet-1K image classification, COCO object detection and ADE20K semantic segmentation.
ImageNet-22K was used for pre-training and ImageNet-1K for fine-tuning.
<br><br>
Here we can see that Swin Transformers achieve great speed-accuracy trade-off compared with the state-of-the-art CNNs.

![exp 1](/media/paper_review_swin_transformer/exp1.png)

Swin Transformers scores better compared with ResNet-50, ResNeXt and DeiT as a backbone for Cascade Mask R-CNN and other detection models.
Also the inference speed is much higher than DeiT's because of linear complexity to input image size.

![exp 2](/media/paper_review_swin_transformer/exp2.png)

Proposed model surpasses other backbones in ADE20K too. 

![exp 3](/media/paper_review_swin_transformer/exp3.png)

Finally, there is an interesting ablation study that shows us that shifted window approach outperforms single window
partitioning with relatively small latency overhead.

![exp 4](/media/paper_review_swin_transformer/exp4.png)

![exp 5](/media/paper_review_swin_transformer/exp5.png)

## Conclusion

I find this paper captivating and useful since it opens new possibilities of developing a unified architecture for
computer vision and natural language processing tasks, which can benefit both fields and accelerate shared research.


