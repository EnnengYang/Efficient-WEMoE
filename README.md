# Code for "Efficient and Effective Weight-Ensembling Mixture of Experts for Multi-Task Model Merging"


<!--

---

## Abstract

> Multi-task learning (MTL) leverages a shared model to accomplish multiple tasks and facilitate knowledge transfer. Recent research on task arithmetic-based MTL demonstrates that merging the parameters of independently fine-tuned models can effectively achieve MTL. However, existing merging methods primarily seek a static optimal solution within the original model parameter space, which often results in performance degradation due to the inherent diversity among tasks and potential interferences. To address this challenge, in this paper, we propose a Weight-Ensembling Mixture of Experts (WEMoE) method for multi-task model merging. Specifically, we first identify critical (or sensitive) modules by analyzing parameter variations in core modules of Transformer-based models before and after finetuning. Then, our WEMoE statically merges non-critical modules while transforming critical modules into a mixture-of-experts (MoE) structure. During inference, expert modules in the MoE are dynamically merged based on input samples, enabling a more flexible and adaptive merging approach. Building on WEMoE, we further introduce an efficient-and-effective WEMoE (E-WEMoE) method, whose core mechanism involves eliminating non-essential elements in the critical modules of WEMoE and implementing shared routing across multiple MoE modules, thereby significantly reducing both the trainable parameters, the overall parameter count, and computational overhead of the merged model by WEMoE. Experimental results across various architectures and tasks demonstrate that both WEMoE and E-WEMoE outperform state-of-the-art (SOTA) model merging methods in terms of MTL performance, generalization, and robustness.

<p align="center">
<img width="885" alt="image" src="https://github.com/user-attachments/assets/c2fbf30d-30a1-4dfa-9e5b-5f5b839e9750">

<img width="470" alt="image" src="https://github.com/user-attachments/assets/2bcb99e5-07e5-4d7a-953b-518c686d79a0">
</p>


[//]: # (## Citation)
[//]: # (If you find our paper or this resource helpful, please consider cite:)
[//]: # (```)
[//]: # (@inproceedings{WEMoE_ICML2024,)
[//]: # (  title={Merging Multi-Task Models via Weight-Ensembling Mixture of Experts},)
[//]: # (  author={Tang, Anke and Shen, Li and Luo, Yong and Yin, Nan and Zhang, Lefei and Tao, Dacheng},)
[//]: # (  booktitle={Forty-first International Conference on Machine Learning},)
[//]: # (  year={2024})
[//]: # (})
[//]: # (```)
[//]: # (Thanks!)

-->

---

## Installation

This project relies on [FusionBench-v0.1.6](https://github.com/tanganke/fusion_bench). Please refer to it to configure the base environment.

```bash
git clone https://github.com/EnnengYang/Efficient-WEMoE
cd Efficient-WEMoE

pip install -e . # install the package in editable mode
```

> [!Note]
> Our code is also integrated into [FusionBench-V0.2.2](https://github.com/tanganke/fusion_bench/tree/main/examples/sparse_we_moe).
> Refer to [https://github.com/tanganke/fusion_bench](https://github.com/tanganke/fusion_bench) for more information.

---

## Run Experiment

- Multi-task performance when merging CLIP-ViT-B/32 or CLIP-ViT-B/16 or CLIP-ViT-L/14 models on all eight tasks
```bash
bash examples/sparse_we_moe/grid_search_sparse_we_moe_ratio.sh
```

- Generalization results on two unseen tasks when merging ViT-B/32 models on six tasks
```bash
bash examples/sparse_we_moe/generalization_vit_b32.sh
```

- Ablations of the test data distribution on ViT-B/32 or CLIP-ViT-B/16 
```bash
bash examples/sparse_we_moe/roubustness.sh
```

*Note: The results of E-WEMoE's experiment can be found in './results/sparse_we_moe/'.*

---

## Acknowledgement
Our implementation references the code below, thanks to them.

- FusionBench: https://github.com/tanganke/fusion_bench

- AdaMerging: https://github.com/EnnengYang/AdaMerging

- Task Arithmetic: https://github.com/mlfoundations/task_vectors

- TIES-MERGING: https://github.com/prateeky2806/ties-merging/tree/main

- Tent: https://github.com/DequanWang/tent
