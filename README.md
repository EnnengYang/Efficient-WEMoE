# Code for "Efficient and Effective Weight-Ensembling Mixture of Experts for Multi-Task Model Merging"
[//]: # ( and Effective Weight-Ensembling Mixture of Experts for Multi-Task Model Merging)

[//]: # (<center>)
[//]: # (<img src="./EWEMoE.png" alt="Efficient Weight-Ensembling MoE" width="800"/>)
[//]: # (</center>)

---

## Installation

This project relies on [FusionBench-v0.1.6](https://github.com/tanganke/fusion_bench). Please refer to it to configure the base environment.

```bash
git clone https://github.com/EnnengYang/Efficient-WEMoE
cd fusion_bench

pip install -e . # install the package in editable mode
```

> [!Note]
> Our code is also integrated into [FusionBench-V0.2.2](https://github.com/tanganke/fusion_bench/tree/main/examples/sparse_we_moe).
> Refer to [https://github.com/tanganke/fusion_bench](https://github.com/tanganke/fusion_bench) for more information.

---

## Train

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
