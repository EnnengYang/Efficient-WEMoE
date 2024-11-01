import functools
import logging
from typing import Optional

from omegaconf import DictConfig, open_dict
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel
from typing_extensions import override

from fusion_bench.dataset import CLIPDataset, load_dataset_from_config
from fusion_bench.utils import timeit_context

from .base_pool import ModelPool
from omegaconf import OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer
from peft import PeftConfig

log = logging.getLogger(__name__)


class PeftHuggingFaceClipVisionPool(ModelPool):
    """
    A model pool for managing Hugging Face's CLIP Vision models.

    This class extends the base `ModelPool` class and overrides its methods to handle
    the specifics of the CLIP Vision models provided by the Hugging Face Transformers library.
    """

    def __init__(self, modelpool_config: DictConfig):
        super().__init__(modelpool_config)

        self._clip_processor = None

    @property
    def clip_processor(self):
        if self._clip_processor is None:
            if "_pretrained_" in self._model_names:
                self._clip_processor = CLIPProcessor.from_pretrained(
                    self.get_model_config("_pretrained_")["path"]
                )
            else:
                log.warning(
                    "No pretrained model found in the model pool. Returning the first model."
                )
                self._clip_processor = CLIPProcessor.from_pretrained(
                    self.get_model_config(self.model_names[0])["path"]
                )
        return self._clip_processor

    def load_model(self, model_config: str | DictConfig) -> CLIPVisionModel:
        """
        Load a CLIP Vision model from the given configuration.

        Args:
            model_config (str | DictConfig): The configuration for the model to load.

        Returns:
            CLIPVisionModel: The loaded CLIP Vision model.
        """
        if isinstance(model_config, str):
            model_config = self.get_model_config(model_config)

        with timeit_context(
            f"Loading CLIP vision model: '{model_config.name}' from '{model_config.path}'."
        ):
            vision_model = CLIPVisionModel.from_pretrained(model_config.path)
        return vision_model

    @override
    def load_merged_lora_model(self, model_config: str | DictConfig) -> CLIPVisionModel:
        """
        Load a CLIP Vision model from the given configuration.

        Args:
            model_config (str | DictConfig): The configuration for the model to load.

        Returns:
            CLIPVisionModel: The loaded CLIP Vision model.
        """
        if isinstance(model_config, str):
            model_config = self.get_model_config(model_config)

        with timeit_context(
            f"Loading LoRA CLIP vision model: '{model_config.name}' from '{model_config.path}'."
        ):
            config = PeftConfig.from_pretrained(model_config["path"])
            clip_model = CLIPVisionModel.from_pretrained(config.base_model_name_or_path)

            clip_model.vision_model = PeftModel.from_pretrained(
                    clip_model.vision_model,
                    model_config["path"],
                    is_trainable=model_config.get("is_trainable", True),
                )

            if model_config.get("merge_and_unload", True):
                clip_model.vision_model.merge_and_unload()
                return clip_model
            else:
                return clip_model

    @override
    def load_unmerged_lora_model(self, model_config: str | DictConfig) -> CLIPVisionModel:
        """
        Load a CLIP Vision model from the given configuration.

        Args:
            model_config (str | DictConfig): The configuration for the model to load.

        Returns:
            CLIPVisionModel: The loaded CLIP Vision model.
        """
        if isinstance(model_config, str):
            model_config = self.get_model_config(model_config)

        with timeit_context(
                f"Loading LoRA CLIP vision model: '{model_config.name}' from '{model_config.path}'."
        ):
            config = PeftConfig.from_pretrained(model_config["path"])
            clip_model = CLIPVisionModel.from_pretrained(config.base_model_name_or_path)

            clip_model.vision_model = PeftModel.from_pretrained(
                clip_model.vision_model,
                model_config["path"],
                is_trainable=model_config.get("is_trainable", True),
            )

            return clip_model
            # return {name: param for name, param in clip_model.named_parameters() if 'lora' in name}

    @override
    def save_model(self, model: CLIPVisionModel, path: str):
        """
        Save a CLIP Vision model to the given path.

        Args:
            model (CLIPVisionModel): The model to save.
            path (str): The path to save the model to.
        """
        with timeit_context(f'Saving clip vision model to "{path}"'):
            model.save_pretrained(path)

    def get_tta_dataset_config(self, dataset: str):
        for dataset_config in self.config.tta_datasets:
            if dataset_config.name == dataset:
                return dataset_config
        raise ValueError(f"Dataset {dataset} not found in config")

    def prepare_dataset_config(self, dataset_config: DictConfig):
        if not hasattr(dataset_config, "type"):
            with open_dict(dataset_config):
                dataset_config["type"] = self.config.dataset_type
        return dataset_config

    @functools.cache
    def get_tta_test_dataset(
        self, tta_dataset: str, clip_processor: Optional[CLIPProcessor] = None
    ):
        """
        Load the test dataset for the task.
        This method is cached, so the dataset is loaded only once.
        """
        if clip_processor is None:
            # if clip_processor is not provided, try to load the clip_processor from pre-trained model
            clip_processor = self.clip_processor
        dataset_config = self.get_tta_dataset_config(tta_dataset)["dataset"]
        dataset_config = self.prepare_dataset_config(dataset_config)
        with timeit_context(f"Loading test dataset: {dataset_config.name}"):
            dataset = load_dataset_from_config(dataset_config)
        dataset = CLIPDataset(dataset, self.clip_processor)
        return dataset

    def get_train_dataset_config(self, model_name: str):
        for dataset_config in self.config.train_datasets:
            if dataset_config.name == model_name:
                return dataset_config
        raise ValueError(f"Dataset {model_name} not found in config")

    def get_train_dataset(
        self, model_name: str, clip_processor: Optional[CLIPProcessor] = None
    ):
        if clip_processor is None:
            # if clip_processor is not provided, try to load the clip_processor from pre-trained model
            clip_processor = self.clip_processor
        dataset_config = self.get_train_dataset_config(model_name)["dataset"]
        dataset_config = self.prepare_dataset_config(dataset_config)
        with timeit_context(f"Loading train dataset: {dataset_config.name}"):
            dataset = load_dataset_from_config(dataset_config)
        dataset = CLIPDataset(dataset, self.clip_processor)
        return dataset
