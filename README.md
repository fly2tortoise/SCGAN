# This paper has just been accepted in April 2025, and I will complete it as soon as possible. 

## Code used for "SCGAN: Sampling and Clustering-based Neural Architecture Search for GANs".

# Introduction
The evolutionary neural architecture search for generative adversarial networks (GANs) has demonstrated promising performance for generating high-quality images. However, two challenges persist, including the long search times and unstable search results. To alleviate these problems, this paper proposes a sampling and clustering-based neural architecture search algorithm for GANs, named SCGAN, which can significantly improve searching efficiency and enhance generation quality. Two improved strategies are proposed in SCGAN. First, a constraint sampling strategy is designed to limit the parameter capacity of architectures, which calculates their architecture size and discards those exceeding a reasonable parameter threshold. Second, a clustering selection strategy is applied in each architecture iteration, which integrates a decomposition selection mechanism and a hierarchical clustering mechanism to further improve search stability. Extensive experiments on the CIFAR-10 and STL-10 datasets demonstrated that SCGAN only requires 0.4 GPU days to find a promising GAN architecture in a vast search space including approximately 10$^{15}$ networks. Our best-found GAN outperformed those obtained by other neural architecture search methods with performance metric results (IS=9.68±0.06, FID=5.54) on CIFAR-10 and (IS=12.12±0.13, FID=12.54) on STL-10.  

# How ot set up SCGAN.  
The environment of SCGAN is more complex, training and searching are torch-based, but part of the evaluation needs to call the api of TensorFlow 2. For better reading, we provide English tutorials and 中文 tutorials.


## 1. Environment requirements:
### 1.1 Basic Requirements
SCGAN's search environment uses the latest version of PyTorch 2.0 and above, along with TensorFlow 2.12 and above.
- python=3.11
- pytorch=2.0.1
- tensorflow=2.12.0
- tensorflow-gan=2.1.0

### 1.2 Baidu Cloud Environment
Considering the difficulty of simultaneously installing and configuring Torch and TensorFlow, we have prepared pre-configured installation packages on Baidu Cloud.

Link: https://pan.baidu.com/s/1I_3zXugfGJAg6l5PEdsV_w
Access Code: 83gs

After downloading, simply extract it to '/home/user/anaconda3/envs/'. The file directory is as follows.
<pre><code>cd /home/yangyeming/anaconda3/envs
tar -xvf torch.tar
</code></pre>
![image](https://github.com/user-attachments/assets/c85ea01b-ac3b-4b81-8fea-a8e990af247b)

Then, activate the relevant environment.
<pre><code>
conda activate torch 
cd SCGAN/search/
</code></pre>

### 1.3 Dataset Preparation (CIFAR-10 and STL-10)
In SCGAN, we use the CIFAR-10 and STL-10 datasets for evaluation. The default datasets are stored in ./datasets/cifar10 and ./datasets/stl10.
Readers can download them manually or use the data code to download them automatically.

### 1.4 Preparing the fid_stat and tmp Folders
You need to download the relevant data from EAGAN. https://github.com/marsggbo/EAGAN
<pre><code>
mkdir fid_stat
mkdir tmp
</code></pre>



## 2. Architecture Search
### 2.1 Constraint Architecture Search to Design GANs
<pre><code>
cd SCGAN/train/
bash train_search_gen.sh
</code></pre> 

### 2.2 Fully Train the Searched GANs Using the MMD-GAN Loss Function
We used the training environment provided by **MMD-AdversarialNAS** and found that the networks trained with the MMD-loss performed well. In this step, you only need to replace the training architecture of MMD-AdversarialNAS with the one found by SCGAN.
<pre><code>
python MGPU_train_arch.py --gpu_ids 0 --num_workers 8 --gen_bs 128 --dis_bs 128 --dataset cifar10 --bottom_width 4 --img_size 32 --max_epoch_G 500 --n_critic 1 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --df_dim 512 --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --val_freq 5 --num_eval_imgs 50000 --exp_name arch_train_cifar10
  
python MGPU_train_arch.py --gpu_ids 0 --num_workers 8 --gen_bs 128 --dis_bs 128 --dataset stl10 --bottom_width 6 --img_size 48 --max_epoch_G 500 --n_critic 1 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --df_dim 512 --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --val_freq 5 --num_eval_imgs 50000 --exp_name arch_train_stl10

bash train_arch_stl10.sh
</code></pre>

## If you have any questions, please email me. I will respond as soon as I have time. Thank you for your attention.

# unfinished and to be continued


# Acknowledgement
Some of the codes are built by:

1.[EAGAN](https://github.com/marsggbo/EAGAN)

2.[AlphaGAN](https://github.com/yuesongtian/AlphaGAN)

3.[Inception Score](https://github.com/openai/improved-gan/tree/master/inception_score) code from OpenAI's Improved GAN (official).

4.[FID Score](https://github.com/bioinf-jku/TTUR) code and CIFAR-10 statistics file from  (official).

5.[MMD-AdversarialNAS](https://github.com/PrasannaPulakurthi/MMD-AdversarialNAS). 

Thanks them for their great works!
