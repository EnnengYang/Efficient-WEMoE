PATH=/home/enneng/fusion_bench/outputs/checkpoints/ViT-B-32/full_finetune_maskadam_lr_0_00001_step_2000

if [ -d "${PATH}/huggingface" ]; then
  echo "${PATH}/huggingface"
else
  mkdir -p "${PATH}/huggingface"
  echo "${PATH}/huggingface"
fi

for TASK in stanford_cars sun397; do
  CUDA_VISIBLE_DEVICES=1 /home/enneng/anaconda3/envs/fusionbench/bin/python convert_checkpoint.py \
  --checkpoint ${PATH}/clip-vit-base-patch32_${TASK} \
  --output ${PATH}/huggingface/clip-vit-base-patch32_${TASK}
done

#step=999
#step_name=1000
#
#for TASK in sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd; do
#  CUDA_VISIBLE_DEVICES=1 /home/enneng/anaconda3/envs/fusionbench/bin/python convert_checkpoint.py \
#  --checkpoint /home/enneng/fusion_bench/outputs/checkpoints/ViT-B-32/full_finetune_lr_0.00001_step_2000/${TASK}/version_0/checkpoints/step=${step}.ckpt \
#  --output /home/enneng/fusion_bench/outputs/checkpoints/ViT-B-32/full_finetune_lr_0.00001_step_2000/huggingface/step_${step_name}/clip-vit-base-patch32_${TASK}
#done