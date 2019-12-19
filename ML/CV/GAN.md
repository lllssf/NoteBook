# GAN

参考资料：

- [简单梳理](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650764594&idx=1&sn=e07c0836dc9a116d71cb899cb6d0cd06&scene=0#wechat_redirecthttps://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650764594&idx=1&sn=e07c0836dc9a116d71cb899cb6d0cd06&scene=0#wechat_redirect)
- [GAN生成图像总数](https://zhuanlan.zhihu.com/p/62746494)

GAN在计算机视觉领域得到了广泛的应用，尤其在图像生成、图像风格迁移、图像标注等方面成果突出。

## 生成模型

生成模型不但可以用于对图像、文本、声音等数据直接建模，还可以用来**建立变量间的条件概率分布**。

除GAN以外，主要的生成模型有：

1. `自回归模型(Autogressive model)`：计算成本高。
2. `变分自编码器(VAE)`：图像模糊。
3. 基于`流(Glow)`的方法：在高分辨率人脸图像合成和插值生成上效果显著。

## Generative Adversarial Networks

GAN是由Goodfellow等人于2014年提出，包含生成器和判别器两个神经网络，生成器将给定的噪声转化为生成图像，判别器分辨生成图像和真实图像，前者的目标是产生判别器难以分辨的图像，后者的目标是更准确地分辨生成图像与真实图像，从而利用博弈的思想优化生成器和判别器使得生成图像与目标图像（真实图像）越来越相近。GAN主要的缺陷在于生成图像的多样性较低，且训练过程不够稳定。

- 论文：
  - [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
  - [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160)
- [code](https://github.com/goodfeli/adversarial) in Github

## DCGAN: 深度卷积生成对抗网络

标准的GAN使用`多层感知机`，而卷积神经网络在处理图像的能力更优，因此使用深度卷积网络改进了GAN，而且使用了`转置卷积操作(Deconvolution)` 使得图像更加清晰生动。

- 论文：[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- [code](https://github.com/floydhub/dcgan) in Github

## CGAN: 条件生成对抗网络

标准的GAN在处理多类目标图像时会生成这些图像模糊的混合图像，因此为了使目标图像分类，CGAN将`one-hot`向量和随机噪声向量拼接。

- 论文：[Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)

## CycleGAN

为了处理图像到图像的翻译问题，CycleGAN应运而生。它包含两个生成器和两个判别器。
![cycleGAN](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gWic8ibtjnE5q7CSQt12WuibaU4TzQrCYdS4eiaIm6rMej6tyajVrNqau8sSh4UNeTWSbkWfBnJNYGqvGg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

- 论文：[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593v6)
- [code](https://github.com/junyanz/CycleGAN) in Github

## CoGAN: 成对(Coupled)对抗生成网络

使用两个共享权重的GAN以期得到更好的效果。
![coGAN](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gWic8ibtjnE5q7CSQt12WuibaU4dTtMJHaUrxecTsA7RkZ9soSmmzsAq5aAd14lfbVfe5soDpEGNsbn8g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

- 论文：[Coupled Generative Adversarial Networks](https://arxiv.org/abs/1606.07536)
- [code](https://github.com/mingyuliutw/CoGAN) in Github

## ProGAN

GAN的训练过程具有不稳定性，导致生成图像优势会出现异常，而ProGAN通过逐层提高生成图像的分辨率来稳定训练过程。

- 论文：[Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)
- [code](https://github.com/tkarras/progressive_growing_of_gans) in Github

## WGAN: Wasserstein生成对抗网络

改变了目标函数，避免了标准GAN可能会出现的梯度消失问题，提高了训练的稳定性。

- 论文：[Wasserstein GAN](https://arxiv.org/abs/1701.07875v3)
- [code](https://github.com/eriklindernoren/Keras-GAN) in Github

## SAGAN: 自注意力生成对抗网络

使得网络可以关注需要注意的特征。

- 论文：[Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318v2)
- [code](https://github.com/heykeetae/Self-Attention-GAN) in Github

## BigGAN：大型生成对抗网络

- 论文：[Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096v2)
- [code](https://github.com/huggingface/pytorch-pretrained-BigGAN) in Github

## StyleGAN: 基于风格的生成对抗网络

应用了`实例归一化(Adaptive instance normalization)，潜在向量映射网络、不断学习的输入`等已有技术。

- 论文：[A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948)
- [code](https://github.com/NVlabs/stylegan) in Github