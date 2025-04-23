# Adaptive Energy Alignment (AEA) for Test-Time Adaptation

Official repository of our work on **Adaptive Energy Alignment (AEA)** for **Test-Time Adaptation (TTA)**. Our paper will be published in **ICLR 2025**. This repository includes a quick summary of our method and a code implementation.

- **Title**: Adaptive Energy Alignment for Accelerating Test-Time Adaptation
- **Full paper link**: https://openreview.net/forum?id=sEMJ1PLSZR

> TL;DR: We propose an adaptive energy alignment (AEA) that accelerates the model adaptation in online TTA scenarios. Specifically, our AEA strategically reduces the energy gap between the source and target domains during TTA, aiming to align the target domain with the source domains and thus to accelerate adaptation.

| ![Image Alt text](/fig/fig1paper.png) | 
|:--:| 
| *Sample-wise logit (x, y-axes) and energy (z-axis) distribution in the early stage of TTA online batches. (a) Without adaptation, target samples are mixed up in the high-energy region, while source samples have low energies and are clearly separated. (b) Our AEA method successfully reduces the energy gap ∆ while accurately guiding the direction of energy alignment. As a result, (c) our approach can accelerate test-time model adaptation and achieve remarkable performance even in a few adaptation batches.* |

## Implementation
Our implementation is build on top of the TTAB benchmark (https://github.com/LINs-lab/ttab.git). \
We provide a command-line script to run our AEA method. \
To get started, please refer to the official TTAB repository for instructions on installing dependencies, setting up datasets, and other prerequisites.

### Quick Run
We provide an example script to help you quickly reproduce our AEA method on the CIFAR10-C dataset with the gaussian_noise severity level 5.

```bash
python run_exp.py \
--job_name aea_tta \
--data_path $PATH_FOR_CIFAR10C$ \
--ckpt_path ./pretrain/ckpt/resnet26_bn_ssh_cifar10.pth \
--model_adaptation_method enetta \
--data_names cifar10_c_deterministic-gaussian_noise-5 \
--loss_name em_energy_sp_wlcs
```


<!-- ## Citing TTAB -->

## Bibliography

```
@inproceedings{zhao2023ttab,
  title     = {On Pitfalls of Test-time Adaptation},
  author    = {Zhao, Hao and Liu, Yuejiang and Alahi, Alexandre and Lin, Tao},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2023},
}
``` 
