# SCGAN
The search code will be published once the paper is accepted, and the training code and network weights will be published immediately.
## Code used for "SCGAN: Sampling and Clustering-based Neural Architecture Search for GANs".

# Introduction
We've desinged a evolutionary neural architecture search algorithm for generative adversarial networks (GANs), dubbed T-EAGAN. Experiments validate the effectiveness of T-EAGAN on the task of unconditional image generation. Extensive experiments on the CIFAR-10 and STL-10 datasets demonstrated that T-EAGAN only requires 1.08 GPU days to find out a superior GAN architecture in a search space including approximately 10<sup>15</sup> network architectures. Our best-found GAN
outperformed those obtained by other neural architecture search methods with performance metric results (IS=9.68±0.06, FID=5.54) on CIFAR-10 and (IS=12.12±0.13, FID=12.54) on STL-10.

# Framework
Fig:framework for SCGAN

# Performance
<!-- 这是注释![](./picture/C10.png)  ![](./picture/S10.png) -->
picture1

picture2

# Set-Up 
## 1.environment requirements:
The search environment is consistent with AlphaGAN，to run this code, you need:  
- PyTorch 1.3.0  
- TensorFlow 1.15.0  
- cuda 10.0  

Other requirements are in environment.yaml 

<!-- install code  -->
<pre><code>conda env create -f environment.yaml
</code></pre>

## 2.prepare fid statistic file
you need to create "fid_stat" directory and download the statistical files of real images.
<pre><code>mkdir fid_stat
</code></pre>

# How to search the  architecture by yourself
## 1. Search on CIFAR-10
<pre><code>bash EAGAN_Only_G30.sh
</code></pre>
# How to train the discovered architecture reported in the paper
## 1. Fully train GAN on CIFAR-10
<pre><code>bash ./scripts/train_arch_cifar10.sh
</code></pre>
## 2. Fully train GAN on STL-10
<pre><code>bash ./scripts/train_arch_stl10.sh
</code></pre>

# How to test the discovered architecture reported in the paper
## 1. Fully train GAN on CIFAR-10
<pre><code>bash ./scripts/test_arch_cifar10.sh
</code></pre>
## 2. Fully train GAN on STL-10
<pre><code>bash ./scripts/test_arch_stl10.sh
</code></pre>

# Acknowledgement
Some of the codes are built by:

1.[EAGAN](https://github.com/marsggbo/EAGAN)

2.[AlphaGAN](https://github.com/yuesongtian/AlphaGAN)

3.[Inception Score](https://github.com/openai/improved-gan/tree/master/inception_score) code from OpenAI's Improved GAN (official).

4.[FID Score](https://github.com/bioinf-jku/TTUR) code and CIFAR-10 statistics file from  (official).

Thanks them for their great works!
