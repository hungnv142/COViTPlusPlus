<h2 align="center"> Find optimal model configuration for COVIDx8A dataset by Multi-objective Optmization approach </h2>

**Abstract.** This work implements One-shot Neural Architecture Search (NAS) to search the best configuration of Pyramid Vision Transformer (PVT) model for COVIDx8A dataset. Specifically, our objective is to find a model which should satisfy the resource constraints: a number of parameters, FLOPs (G) while also guaranteeing performance metrics: accuracy on test set and COVID-19 sensitivity. In order to tackle these challenges, we construct the large search space covering the changeable dimensions of Multi-head Attention vectors (*Q-K-V*); changeable pooling dimensions to perform Linear Spatial Reduction Attention; MLP ratios and amount of Transformer Encoders of each Stage. We initially train the supernet (the model can cover all scenarios from the search spacce) by Knowledge Distillation stragedy with the help of [Exp.5 teacher model](https://github.com/hoangtv2000/COViT#the-exp5-model). Benefitting the **weight-entanglement training stragedy** of Vision Transformers, the well-trained supernet can allow a thousand of its subnets to be well-trained without having to train from scratch. Employing the **One-shot NAS searching technique**, our best subnet achieves 88.54% top-1 accuracy and 90.65% sensitivity on COVIDx8A dataset, with 3.16 M parameters and 0.29 GFLOPs in the retraining phase.

## 1. What is One-shot NAS?
+ A one-shot model (**supernet**) contains all possible architectures in subspace as submodels (**subnets**).
+ It allows the weights are shared between different architectures with common layers in supernet.
+ We only have to train the single one-shot model => search costs are reduced drastically.


## 2. What is Weight-entanglement stragedy for One-shot NAS?
The central idea of weight-entanglement stragedy is to enable different transformer blocks to share weights for their common parts in each layer. In particular, the weight entanglement strategy enforces that different candidate blocks in the same layer to share as many weights as possible. Thus the training of any block will affect the weights of others for their intersected portion, as demonstrated in the figure below. During implementation, for each layer, we need to store only the weights of the largest block among the homogeneous candidates. The remaining smaller building blocks can directly extract weights from the largest one. Note that the proposed weight entanglement strategy is dedicated to work on homogeneous building blocks, such as self-attention modules. The Depth-wise Convolutional blocks can not inherit this property, so we devide into 2 scenarios of MLP ratios, and training with individual 2 supernets.

<p align="center">
	<img src="https://user-images.githubusercontent.com/58163069/157636260-10f84efd-0bd4-4319-b625-c1b6d96cd054.png" alt="photo not available" width="65%" height="65%">
	<br>
	<em>Comparison between Classical weight sharing and Weight Entanglement</em>
</p>


## 3. Search-space configuration

|QKV Embed. dims     |  MLP ratios       | Depths        | Sample pool. dims |
| :----------------: | :---------------: | :-----------: | :---------------: |
| [8, 16, 40, 64]    | [8, 8, 8, 4]      | [2, 2, 2, 2]  | 7                 |
| [16, 32, 80, 128]  | [8, 8, 4, 4]      | [2, 2, 4, 2]  | 15                |
| [24, 48, 120, 192] |                   | [2, 3, 6, 2]  | 31                |
| [32, 64, 160, 256] |                   | [2, 4, 8, 2]  |                   |
|                    |                   | [3, 4, 12, 2] |                   |

+ Params range: 2.7 - 9.1 M
+ Flops range: 0.4 - 10.6 G
+ Number of available sub-model: 120 models

## 4. Pareto frontier of Test Accuracy, COVID-19 Sensitivity vs. Params, GFLOPs    
<p align="center">
	<img src="https://user-images.githubusercontent.com/58163069/157635984-6bc97aee-4e6b-4a0d-88cc-a5186fd03f8e.png" alt="photo not available" width="75%" height="75%">
	<br>
	<em>Top-1 accuracy and COVID-19 sensitivity on COVIDx8A dataset and top 100 sampled high-performing architectures from the supernet with weight inherited from the supernet.</em>
</p>


### 4.1. The most efficient model configuration

|QKV Embed. dims     |  MLP ratios       | Depths        | Sample pool. dims |  Number of params  |  FLOPS             |
| :----------------: | :---------------: | :-----------: | :---------------: |  :---------------: |  :---------------: |
| [16, 32, 80, 128]  | [8, 8, 8, 4]      | [2, 2, 2, 2]  | 7                 | 3.16 M             | 0.29 G             |

### 4.2. Retrained model weights
[Click here](https://drive.google.com/file/d/18JERejTOmcCAUTjSZZzJmkKsb7Kpf6aW/view?usp=sharing) to download retrained model weights.


### 4.3. Experimental results
|Acc. (%)   	  |  Sens. Pneu.      |Sens. Normal   |Sens. COVID-19   | PPV Pneu.        | PPV Norm.  | PPV COVID-19  | Macro Avg. F1 (%)   |
| :-------------: | :---------------: | :-----------: | :-------------: |  :-------------: |  :-------: |  :----------: |  :----------------: |
| 88.54           | 86.67 	      |86.0 	      |90.65		| 82.72 	   | 83.50 	| 94.17 	|  87.25              |



## 5. Future works 
Despite creating a large search space including various configurations to find out the most compact and effective model for COVIDx8A dataset. We still hand-craftedly study the scaling factor of Depthwise Convolutional layers (MLP-ratios) according to the base configurations of PVT's original models by creating the two independent supernets. We continue to study the well-performing Convolutional layers searching method, combining with the Weight Entanglement method to create a unique supernet.

## Reference
[1] W. Wang, E. Xie, X. Li, D.-P. Fan, K. Song, D. Liang, T. Lu, P. Luo, and L. Shao, “PVTv2: Improved
Baselines with Pyramid Vision Transformer,” arXiv e-prints, p. arXiv:2106.13797, Jun. 2021.

[2] Minghao Chen, Houwen Peng, Jianlong Fu, and Haibin Ling. Autoformer: Searching transformers for
visual recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV),
2021.

[3] https://github.com/microsoft/Cream
