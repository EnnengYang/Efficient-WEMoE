import functools
import logging
import os
from abc import abstractmethod
from copy import deepcopy
from typing import List

import lightning as L
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel
from transformers.models.clip.modeling_clip import CLIPEncoder, CLIPEncoderLayer

from fusion_bench.method.base_algorithm import ModelFusionAlgorithm
from fusion_bench.method.task_arithmetic import task_arithmetic_merge
from fusion_bench.modelpool import ModelPool
from fusion_bench.modelpool.huggingface_clip_vision import HuggingFaceClipVisionPool
from fusion_bench.models.hf_clip import HFCLIPClassifier
from fusion_bench.models.sparse_we_moe import SparseWeightEnsemblingMoE
from fusion_bench.models.sparse_we_moe import SparseWeightEnsemblingMoE_ShardGate
from fusion_bench.models.sparse_we_moe import construct_weight_ensembling_gate
from fusion_bench.tasks.clip_classification import get_classnames_and_templates
from fusion_bench.utils import timeit_context
from fusion_bench.utils.data import InfiniteDataLoader

from .sparse_we_moe import SparseWeightEnsemblingMoEAlgorithm
from tqdm.auto import tqdm
import numpy as np

log = logging.getLogger(__name__)

class SparseCLIPWeightEnsemblingMoEAlgorithm(SparseWeightEnsemblingMoEAlgorithm):
    modelpool: HuggingFaceClipVisionPool = None
    _clip_processor: CLIPProcessor = None
    zeroshot_weights = {}

    def load_checkpoint(self, model, checkpoint):
        state = {"model": model}
        self._fabric.load(checkpoint, state)

    def save_checkpoint(self, model, checkpoint):
        self._fabric.save(checkpoint, {"model": model})

    def construct_moe_model(self) -> SparseWeightEnsemblingMoE:
        base_model = self.modelpool.load_model("_pretrained_")
        expert_models = [
            self.modelpool.load_model(m) for m in self.modelpool.model_names
        ]

        # merge the models using task arithmetic
        moe_model = task_arithmetic_merge(
            # this function modifies the model in place, so we need to pass a deepcopy
            deepcopy(base_model),
            expert_models,
            scaling_factor=self.config.init_lambda,
        ).requires_grad_(False)

        # up-scale MLP modules
        base_encoder: CLIPEncoder = base_model.vision_model.encoder
        moe_encoder: CLIPEncoder = moe_model.vision_model.encoder
        expert_encoders = [m.vision_model.encoder for m in expert_models]

        num_layers = len(base_encoder.layers)
        for layer_idx in range(num_layers):
            base_mlp = base_encoder.layers[layer_idx].mlp
            expert_mlps = [e.layers[layer_idx].mlp for e in expert_encoders]

            moe_encoder.layers[layer_idx].mlp = SparseWeightEnsemblingMoE(
                hidden_size=base_encoder.config.hidden_size,
                base_model=base_mlp,
                expert_models=expert_mlps,
                init_lambda=self.config.init_lambda,
                batch_first=True,  # for open_clip models this is False
                router_hidden_layers=self.config.router_hidden_layers,
                batch_reduce=self.config.batch_reduce,
                num_layers=num_layers,
                layer_idx=layer_idx,
                tv_prune_ratio=self.config.tv_prune_ratio,
            )

        return moe_model
    def construct_moe_model_sharedgate(self) -> SparseWeightEnsemblingMoE_ShardGate:
        base_model = self.modelpool.load_model("_pretrained_")
        expert_models = [
            self.modelpool.load_model(m) for m in self.modelpool.model_names
        ]

        # merge the models using task arithmetic
        moe_model = task_arithmetic_merge(
            # this function modifies the model in place, so we need to pass a deepcopy
            deepcopy(base_model),
            expert_models,
            scaling_factor=self.config.init_lambda,
        ).requires_grad_(False)

        # up-scale MLP modules
        base_encoder: CLIPEncoder = base_model.vision_model.encoder
        moe_encoder: CLIPEncoder = moe_model.vision_model.encoder
        expert_encoders = [m.vision_model.encoder for m in expert_models]

        # shared gate
        shared_gate = construct_weight_ensembling_gate(
            hidden_size = base_encoder.config.hidden_size + self.config.position_encoding_dim
                          if self.config.position_encoding else base_encoder.config.hidden_size,
            num_experts=len(expert_models),
            init_lambda=self.config.init_lambda,
            num_hidden_layers=self.config.router_hidden_layers,
        )


        # ------------------------------------------------------------------------------------
        # Calculate magnitude
        # num_layers = len(base_encoder.layers)
        # exp_id = 0
        # for e in expert_encoders:
        #     for layer_idx in range(num_layers):
        #         if layer_idx in [0,3,5,7,9,11]:
        #             print(f"layer_idx: {layer_idx}")
        #             v_e = torch.cat([param.view(-1) for param in e.layers[layer_idx].mlp.parameters()])
        #             v_base = torch.cat([param.view(-1) for param in base_encoder.layers[layer_idx].mlp.parameters()])
        #             absolute_vector = torch.abs(v_e - v_base)
        #             np.save(f"/home/enneng/fusion_bench/outputs/sparse_we_moe/magnitude/absolute_vector_expert_{exp_id}_layer_{layer_idx}.npy", absolute_vector.detach().numpy())
        #     exp_id += 1
        # print('succ')
        # ------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------
        # Calculate l2 distance and cos similarity
        # key = 'att' # 'mlp' or 'att'
        # num_layers = len(base_encoder.layers)
        # l2_distance_ss = []
        # cos_sim_ss = []
        # for e in expert_encoders:
        #     l2_distance_s = []
        #     cos_sim_s = []
        #     for layer_idx in range(num_layers):
        #         print(f"layer_idx: {layer_idx}")
        #         v_e = torch.cat([param.view(-1) for param in e.layers[layer_idx].mlp.parameters()]) if key == 'mlp' \
        #             else torch.cat([param.view(-1) for param in e.layers[layer_idx].self_attn.parameters()])
        #         v_base = torch.cat([param.view(-1) for param in base_encoder.layers[layer_idx].mlp.parameters()]) if key == 'mlp' \
        #             else torch.cat([param.view(-1) for param in base_encoder.layers[layer_idx].self_attn.parameters()])
        #         l2_distance = torch.norm(v_e - v_base, p=2)
        #         print(f"L2 Distance: {l2_distance}")
        #         cos_sim = torch.nn.functional.cosine_similarity(v_e, v_base, dim=0)
        #         print(f"Cosine Similarity: {cos_sim}")
        #
        #         l2_distance_s.append(l2_distance.item())
        #         cos_sim_s.append(cos_sim.item())
        #     l2_distance_ss.append(l2_distance_s)
        #     cos_sim_ss.append(cos_sim_s)
        #
        # print("L2 Distances:")
        # print(l2_distance_ss)
        # print("Cosine Similarity:")
        # print(cos_sim_ss)
        # ------------------------------------------------------------------------------------


        num_layers = len(base_encoder.layers)
        for layer_idx in range(num_layers):
            base_mlp = base_encoder.layers[layer_idx].mlp
            expert_mlps = [e.layers[layer_idx].mlp for e in expert_encoders]

            moe_encoder.layers[layer_idx].mlp = SparseWeightEnsemblingMoE_ShardGate(
                hidden_size=base_encoder.config.hidden_size,
                base_model=base_mlp,
                expert_models=expert_mlps,
                init_lambda=self.config.init_lambda,
                batch_first=True,  # for open_clip models this is False
                router_hidden_layers=self.config.router_hidden_layers,
                batch_reduce=self.config.batch_reduce,
                num_layers=num_layers,
                layer_idx=layer_idx,
                tv_prune_ratio=self.config.tv_prune_ratio,
                sharedgate=shared_gate,
                position_encoding=self.config.position_encoding,
                position_encoding_dim=self.config.position_encoding_dim,
            )

        return moe_model

    @functools.cache
    def get_shuffled_test_loader_iter(self, tta_dataset: str):
        log.info("get_shuffled_test_loader_iter")
        loader = DataLoader(
            self.modelpool.get_tta_test_dataset(
                tta_dataset, clip_processor=self._clip_processor
            ),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        if self._fabric is not None:
            loader = self._fabric.setup_dataloaders(loader)
        return iter(InfiniteDataLoader(loader))

    def on_test_time_adaptation_start(self):
        """
        Here we load the CLIP processor and construct the zero-shot classification head for each task.
        """
        clip_model_config = self.modelpool.get_model_config("_pretrained_")

        with timeit_context("Loading CLIP processor and pretrained CLIP model."):
            self._clip_processor = CLIPProcessor.from_pretrained(clip_model_config.path)
            clip_model = CLIPModel.from_pretrained(clip_model_config.path)

            clip_classifier = HFCLIPClassifier(clip_model, self._clip_processor)
            self.visual_projection = clip_model.visual_projection.requires_grad_(False)
            self.logit_scale = clip_model.logit_scale.exp()
            if self._fabric is not None:
                self.visual_projection = self._fabric.to_device(self.visual_projection)
                self.logit_scale = self._fabric.to_device(self.logit_scale)

        for task in self.modelpool.model_names:
            cache_file = os.path.join(
                self.config.cache_dir,
                f"{os.path.basename(clip_model_config.path)}_{task}_zeroshot_weights.pt",
            )
            if os.path.exists(cache_file):
                log.info(f"Loading cached zeroshot weights for task: {task}")
                zeroshot_weights = torch.load(cache_file, map_location="cpu")
            else:
                log.info(f"Construct zero shot classification head for task: {task}")
                classnames, templates = get_classnames_and_templates(
                    self.modelpool.get_tta_dataset_config(task)["dataset"].name
                )
                clip_classifier.set_classification_task(classnames, templates)
                zeroshot_weights = clip_classifier.zeroshot_weights
                log.info(f"save zeroshot weights to {cache_file}")
                torch.save(zeroshot_weights, cache_file)
            self.zeroshot_weights[task] = zeroshot_weights
            if self._fabric is not None:
                self.zeroshot_weights[task] = self._fabric.to_device(
                    self.zeroshot_weights[task]
                )

    def compute_logits(self, module, batch, task) -> Tensor:
        images, _ = batch
        text_embeds = self.zeroshot_weights[task]

        image_embeds = module(images)[1]
        image_embeds = self.visual_projection(image_embeds)

        # normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * self.logit_scale
        logits_per_image = logits_per_text.t()

        return logits_per_image
