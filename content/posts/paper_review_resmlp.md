---
title: "Paper Review: ResMLP: Feedforward networks for image classification with data-efficient training"
date: "2021-05-11T23:46:37.121Z"
template: "post"
draft: false
slug: "paper-review-resmlp"
category: "Paper Review"
tags:
  - "Computer Vision"
  - "Image Classification"
description: "My review of the paper ResMLP: Feedforward networks for image classification with data-efficient training"
socialImage: "/media/paper_review_resmlp/collage.png"
---

[Paper link](https://arxiv.org/abs/2105.03404)

Code available [here](https://github.com/lucidrains/res-mlp-pytorch) 

![collage](/media/paper_review_resmlp/collage.png)

In this paper, researchers from Facebook AI, Sorbonne University and INRIA present ResMLP, an architecture built entirely upon multi-layer percep-trons for image classification. It's a residual network which that contains a linear layer in which patches 
interact independently across channels and a two-layer feed-forward network in which channels interact independently per patch.
When trained with modern training strategies it shows decent accuracy / complexity tradeoff on ImageNet.

## Method   

![fig 1](/media/paper_review_resmlp/fig_1.png)

#### The Residual Multi-Layer Perceptron

Residual Multi-Layer Perceptron is a combination of linear and feedforward layers paralleled by a skip-connection. 
Authors replace Layer Normalization with simple affine transformation because of sufficiently stable training. It's used twice
for each block - as a pre-normalization and as a post-processing to replicate behaviour similar to the recent
LayerScale which improves the optimization of deep transformers.

![affine](/media/paper_review_resmlp/affine.png)

Also ReLU replaced by a GeLU function. Self-attention replaced by a linear interaction.
<br><br>
Overall structure of the perceptron

![mlp_block](/media/paper_review_resmlp/mlp_block.png)

#### Overall idea

1. Take *N x N* non-overlapping patches as input. 
2. Create a set of *N squared* d-dimensional embeddings.
3. Feed embeddings to several Residual Multi-Layer Perceptron layers to produce a set of *N squared*.
d-dimensional output embeddings.
4. Average embeddings as a d-dimensional vector using average pooling or an alternative of class-attention introduced in CaiT paper.
5. Feed obtained vector to a linear classifier.

## Experiments

Models were trained on the ImageNet-1K dataset.

#### ImageNet classification

![experiments](/media/paper_review_resmlp/experiments.png)

#### Transfer learning

![transfer](/media/paper_review_resmlp/transfer.png)

Original paper have interesting ablation study which refers to communication with low-resolution
convolutions and normalization.

## Conclusion

Authors pointed out that simple residual architecture combined with modern training strategy 
can achieve high perfomance on image classification task. Work will contribute to understanding 
of designing networks without pyramidal structure adopted by most convolutional neural networks.
