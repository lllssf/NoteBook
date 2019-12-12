# Deep Image Prior 阅读笔记

## Abstract

1. **`生成网络`的结构足够捕捉低级的图像先验信息。**
2. 随机初始化的神经网络在诸如去噪、超分辨和图像修复等反演任务上用于学习先验知识非常好的效果。

## Introduction

1. 深层卷积神经网络（`ConvNets`）在图像重构领域目前处于领先地位（state-of-the-art）。
2. 具有类似结构的ConvNets像是`GAN（generative adversarial networks）`、`变分自编码器（variational autoencoders）`和 direct pixelwise error minimization 被用于图像生成领域。
3. 但这些领先的ConvNets总是依靠大训练集。因此其优越的性能被推测是由于其从数据中习得图像先验知识的能力。但是有文章表明网络的结构同样重要。
4. 本文提出图像生成网络本身的结构就可以获取大量低等级图像先验信息，并使用未训练的ConvNets来解决图像重构问题。
5. 将生成网络与有损的图像匹配，其网络参数可以用于图像重构。权重是随机初始化，被训练至最大化降质图像与任务目标的相似度。
6. *No aspect of the network is learned from data*，先验知识来源于网络的结构本身。

## Method
1. $x = f_\theta(z)$
2. 在这篇文章中，为了表明未训练的生成网络可以捕捉先验知识，$x$指目标图片，$z$是随机的张量/向量，$\theta$是未训练的生成网络的参数。
3. 使用的`encoder-decoder`的“沙漏”结构。
4. 重构任务表征为： $x^* = minE(x;x_0) + R(x)$
5. 正则项$R(x)$的选择在很多论文中都是研究重点，因为不同的选择会捕捉到不同的一般先验信息。在本文中，使用生成网络捕捉的隐形先验信息代替正则项。
6. $\theta^* = argminE(f_\theta(z);x_0)$ ，$x^* = f_\theta^*(z)$
7. $x_0$是低质图像，最小的$\theta^*$是从随即参数使用诸如SGD的优化算法得到的。
8. 得到（局部）最优的$\theta^*$，就可以通过$x^* = f_\theta^*(z)$恢复图像。一般也可以优化$z$，但本文是将它*随机初始化并保持不变*。对于由$z$经特定结构的网络可以生成的图片，$R(x)=0$；对于其他信号，$R(x)=+\infty$。
9. 
