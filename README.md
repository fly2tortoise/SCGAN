# T-EAGAN

## Code used for  Time-constrained Evolutionary Neural Architecture Search for Generative Adversarial Networks.

# Introduction
We've desinged a evolutionary neural architecture search algorithm for generative adversarial networks (GANs), dubbed T-EAGAN. Experiments validate the effectiveness of T-EAGAN on the task of unconditional image generation. Extensive experiments on the CIFAR-10 and STL-10 datasets demonstrated that T-EAGAN only requires 2 GPU days to find out a superior GAN architecture in a search space including approximately 10^15 network architectures. Our best found network architectures could outperform those obtained by other neural architecture search methods with the performance metric results(IS=8.957±0.08, FID=9.432) on CIFAR-10 and (IS=10.576±0.085, FID=20.323) on STL-10.

## framework
Fig:framework for T-EAGAN

# performance

Fig:CIFAR-10

Fig:STL-10

# Set-Up 
## environment requirements:

## prepare fid statistic file


# How to search & train the derived architecture by yourself


# How to train & test the discovered architecture reported in the paper
