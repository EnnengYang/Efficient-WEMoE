# clip-vit-base-patch16

CUDA_VISIBLE_DEVICES=1 fusion_bench \
    method=weight_ensembling_moe \
    method.name=clip_weight_ensembling_moe \
    method.use_grad_accumulate=false \
    fast_dev_run=false \
    modelpool=clip-vit-base-patch16_TA8 \
    taskpool=clip-vit-classification_TA8_B16\
    save_report=outputs/clip-vit-base-patch16/clip-vit-base-patch16-weight_ensembling_moe.json

CUDA_VISIBLE_DEVICES=1 fusion_bench \
    method=adamerging \
    method.name=clip_layer_wise_adamerging  \
    fast_dev_run=false \
    modelpool=clip-vit-base-patch16_TA8 \
    taskpool=clip-vit-classification_TA8_B16\
    save_report=outputs/clip-vit-base-patch16/clip-vit-base-patch16-clip_layer_wise_adamerging.json

CUDA_VISIBLE_DEVICES=1 fusion_bench \
    method=ties_merging  \
    method.scaling_factor=0.3\
    method.threshold=20 \
    modelpool=clip-vit-base-patch16_TA8 \
    taskpool=clip-vit-classification_TA8_B16\
    save_report=outputs/clip-vit-base-patch16/clip-vit-base-patch16-clip_ties_merging.json

CUDA_VISIBLE_DEVICES=1 fusion_bench \
    method=task_arithmetic   \
    method.scaling_factor=0.3\
    modelpool=clip-vit-base-patch16_TA8 \
    taskpool=clip-vit-classification_TA8_B16\
    save_report=outputs/clip-vit-base-patch16/clip-vit-base-patch16-clip_task_arithmetic.json

CUDA_VISIBLE_DEVICES=1 fusion_bench \
    method=simple_average    \
    modelpool=clip-vit-base-patch16_TA8 \
    taskpool=clip-vit-classification_TA8_B16\
    save_report=outputs/clip-vit-base-patch16/clip-vit-base-patch16-simple_average.json

CUDA_VISIBLE_DEVICES=1 fusion_bench \
    method=clip_regmean   \
    modelpool=clip-vit-base-patch16_TA8 \
    taskpool=clip-vit-classification_TA8_B16\
    save_report=outputs/clip-vit-base-patch16/clip-vit-base-patch16-clip_regmean.json

CUDA_VISIBLE_DEVICES=1 fusion_bench \
    method=clip_fisher_merging   \
    modelpool=clip-vit-base-patch16_TA8 \
    taskpool=clip-vit-classification_TA8_B16\
    save_report=outputs/clip-vit-base-patch16/clip-vit-base-patch16-fisher_merging.json

CUDA_VISIBLE_DEVICES=1 fusion_bench \
    method=dummy     \
    modelpool=clip-vit-base-patch16_TA8 \
    modelpool.models.0.path=openai/clip-vit-base-patch16 \
    taskpool=clip-vit-classification_TA8_B16\
    save_report=outputs/clip-vit-base-patch16/clip-vit-base-patch16-pretrain.json