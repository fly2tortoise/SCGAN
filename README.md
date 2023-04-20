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
## environment requirements:
The search environment is consistent with AlphaGAN，to run this code, you need:  
- PyTorch 1.3.0  
- TensorFlow 1.15.0  
- cuda 10.0  

Other requirements are in environment.yaml 

<!-- install code  -->
<pre><code>conda env create -f environment.yaml
</code></pre>

## prepare fid statistic file


# How to search the  architecture by yourself


# How to train the discovered architecture reported in the paper


# How to test the discovered architecture reported in the paper
