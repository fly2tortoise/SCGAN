# T-EAGAN

## Code used for "Time-constrained Evolutionary Neural Architecture Search for Generative Adversarial Networks".

# Introduction
We've desinged a evolutionary neural architecture search algorithm for generative adversarial networks (GANs), dubbed T-EAGAN. Experiments validate the effectiveness of T-EAGAN on the task of unconditional image generation. Extensive experiments on the CIFAR-10 and STL-10 datasets demonstrated that T-EAGAN only requires 1.08 GPU days to find out a superior GAN architecture in a search space including approximately 10<sup>15</sup> network architectures. Our best found network architectures could outperform those obtained by other neural architecture search methods with the performance metric results(**IS=8.957±0.08, FID=9.432**) on CIFAR-10 and (**IS=10.576±0.085, FID=20.323**) on STL-10.

# Framework
Fig:framework for T-EAGAN

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


