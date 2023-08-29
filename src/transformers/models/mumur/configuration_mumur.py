# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" MuMUR model configuration"""

import os
from typing import Union

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

MUMUR_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Intel/mumur": "https://huggingface.co/Intel/mumur/resolve/main/config.json",
}



class MuMURTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MuMURModel`]. It is used to instantiate an
    MuMUR model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the MuMUR
    [CIDAS/mumur-rd64](https://huggingface.co/CIDAS/mumur-rd64) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 49408):
            Vocabulary size of the MuMUR text model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`MuMURModel`].
        hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        max_position_embeddings (`int`, *optional*, defaults to 77):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float``, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from transformers import MuMURTextConfig, MuMURTextModel

    >>> # Initializing a MuMURTextConfig with CIDAS/mumur-rd64 style configuration
    >>> configuration = MuMURTextConfig()

    >>> # Initializing a MuMURTextModel (with random weights) from the CIDAS/mumur-rd64 style configuration
    >>> model = MuMURTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "mumur_text_model"

    def __init__(
        self,
        vocab_size=49408,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=8,
        max_position_embeddings=77,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=49406,
        eos_token_id=49407,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the text config dict if we are loading from MuMURConfig
        if config_dict.get("model_type") == "mumur":
            config_dict = config_dict["text_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)

class MuMURMultilingualTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MuMURModel`]. It is used to instantiate an
    MuMUR model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the MuMUR
    [CIDAS/mumur-rd64](https://huggingface.co/CIDAS/mumur-rd64) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 49408):
            Vocabulary size of the MuMUR text model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`MuMURModel`].
        hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        max_position_embeddings (`int`, *optional*, defaults to 77):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float``, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from transformers import MuMURTextConfig, MuMURTextModel

    >>> # Initializing a MuMURTextConfig with CIDAS/mumur-rd64 style configuration
    >>> configuration = MuMURTextConfig()

    >>> # Initializing a MuMURTextModel (with random weights) from the CIDAS/mumur-rd64 style configuration
    >>> model = MuMURTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "mumur_multilingual_text_model" #m-clip

    def __init__(
        self,
        # vocab_size=49408,
        hidden_size=1024,
        # intermediate_size=2048,
        # num_hidden_layers=12,
        # num_attention_heads=8,
        # max_position_embeddings=77,
        # hidden_act="quick_gelu",
        # layer_norm_eps=1e-5,
        # attention_dropout=0.0,
        # initializer_range=0.02,
        # initializer_factor=1.0,
        # pad_token_id=1,
        # bos_token_id=49406,
        # eos_token_id=49407,
        model_name='xlm-roberta-large',
        output_dim=512,
        **kwargs,
        ):
        # super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        super().__init__(hidden_size=hidden_size, model_name=model_name, output_dim=output_dim, **kwargs)
        self.hidden_size = hidden_size
        self.model_name = model_name
        self.output_dim = output_dim

    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
    #     cls._set_token_in_kwargs(kwargs)

    #     config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

    #     # get the text config dict if we are loading from MuMURConfig
    #     if config_dict.get("model_type") == "mumur":
    #         config_dict = config_dict["text_config"]

    #     if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
    #         logger.warning(
    #             f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
    #             f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
    #         )

    #     return cls.from_dict(config_dict, **kwargs)
    
class MuMURVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MuMURModel`]. It is used to instantiate an
    MuMUR model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the MuMUR
    [CIDAS/mumur-rd64](https://huggingface.co/CIDAS/mumur-rd64) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float``, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from transformers import MuMURVisionConfig, MuMURVisionModel

    >>> # Initializing a MuMURVisionConfig with CIDAS/mumur-rd64 style configuration
    >>> configuration = MuMURVisionConfig()

    >>> # Initializing a MuMURVisionModel (with random weights) from the CIDAS/mumur-rd64 style configuration
    >>> model = MuMURVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mumur_vision_model"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=32,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        num_frames=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.num_frames = num_frames

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from MuMURConfig
        if config_dict.get("model_type") == "mumur":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)

class MuMURMultiModalConfig(PretrainedConfig):
    
    def __init__(
        self,
        hidden_size=512,
        num_attention_heads=1,
        attention_dropout=0.0,
        output_dim=512, #TODO check size
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        # self.intermediate_size = intermediate_size
        # self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        # self.num_channels = num_channels
        # self.patch_size = patch_size
        # self.image_size = image_size
        # self.initializer_range = initializer_range
        # self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        # self.layer_norm_eps = layer_norm_eps
        # self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.output_dim = output_dim

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from MuMURConfig
        if config_dict.get("model_type") == "mumur":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)

class MuMURConfig(PretrainedConfig):
    r"""
    [`MuMURConfig`] is the configuration class to store the configuration of a [`MuMURModel`]. It is used to
    instantiate a MuMUR model according to the specified arguments, defining the text model and vision model configs.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the MuMUR
    [CIDAS/mumur-rd64](https://huggingface.co/CIDAS/mumur-rd64) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`MuMURTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`MuMURVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* paramter. Default is used as per the original MuMUR implementation.
        extract_layers (`List[int]`, *optional*, defaults to [3, 6, 9]):
            Layers to extract when forwarding the query image through the frozen visual backbone of CLIP.
        reduce_dim (`int`, *optional*, defaults to 64):
            Dimensionality to reduce the CLIP vision embedding.
        decoder_num_attention_heads (`int`, *optional*, defaults to 4):
            Number of attention heads in the decoder of MuMUR.
        decoder_attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        decoder_hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        decoder_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layers in the Transformer decoder.
        conditional_layer (`int`, *optional*, defaults to 0):
            The layer to use of the Transformer encoder whose activations will be combined with the condition
            embeddings using FiLM (Feature-wise Linear Modulation). If 0, the last layer is used.
        use_complex_transposed_convolution (`bool`, *optional*, defaults to `False`):
            Whether to use a more complex transposed convolution in the decoder, enabling more fine-grained
            segmentation.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import MuMURConfig, MuMURModel

    >>> # Initializing a MuMURConfig with CIDAS/mumur-rd64 style configuration
    >>> configuration = MuMURConfig()

    >>> # Initializing a MuMURModel (with random weights) from the CIDAS/mumur-rd64 style configuration
    >>> model = MuMURModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a MuMURConfig from a MuMURTextConfig and a MuMURVisionConfig

    >>> # Initializing a MuMURText and MuMURVision configuration
    >>> config_text = MuMURTextConfig()
    >>> config_vision = MuMURVisionConfig()

    >>> config = MuMURConfig.from_text_vision_configs(config_text, config_vision)
    ```"""

    model_type = "mumur"

    def __init__(
        self,
        text_config=None,
        multilingual_text_config=None,
        vision_config=None,
        multimodal_config=None,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        extract_layers=[3, 6, 9],
        reduce_dim=64,
        decoder_num_attention_heads=4,
        decoder_attention_dropout=0.0,
        decoder_hidden_act="quick_gelu",
        decoder_intermediate_size=2048,
        conditional_layer=0,
        use_complex_transposed_convolution=False,
        **kwargs,
    ):
        # If `_config_dict` exist, we use them for the backward compatibility.
        # We pop out these 2 attributes before calling `super().__init__` to avoid them being saved (which causes a lot
        # of confusion!).
        text_config_dict = kwargs.pop("text_config_dict", None)
        vision_config_dict = kwargs.pop("vision_config_dict", None)
        multilingual_text_config_dict = kwargs.pop("multilingual_text_config_dict", None)
        multimodal_config_dict = kwargs.pop("multimodal_config", None)

        super().__init__(**kwargs)

        # Instead of simply assigning `[text|vision]_config_dict` to `[text|vision]_config`, we use the values in
        # `[text|vision]_config_dict` to update the values in `[text|vision]_config`. The values should be same in most
        # cases, but we don't want to break anything regarding `_config_dict` that existed before commit `8827e1b2`.
        if text_config_dict is not None:
            if text_config is None:
                text_config = {}

            # This is the complete result when using `text_config_dict`.
            _text_config_dict = MuMURTextConfig(**text_config_dict).to_dict()

            # Give a warning if the values exist in both `_text_config_dict` and `text_config` but being different.
            for key, value in _text_config_dict.items():
                if key in text_config and value != text_config[key] and key not in ["transformers_version"]:
                    # If specified in `text_config_dict`
                    if key in text_config_dict:
                        message = (
                            f"`{key}` is found in both `text_config_dict` and `text_config` but with different values. "
                            f'The value `text_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`text_config_dict` is provided which will be used to initialize `MuMURTextConfig`. The "
                            f'value `text_config["{key}"]` will be overriden.'
                        )
                    logger.warning(message)

            # Update all values in `text_config` with the ones in `_text_config_dict`.
            text_config.update(_text_config_dict)

        if vision_config_dict is not None:
            if vision_config is None:
                vision_config = {}

            # This is the complete result when using `vision_config_dict`.
            _vision_config_dict = MuMURVisionConfig(**vision_config_dict).to_dict()
            # convert keys to string instead of integer
            if "id2label" in _vision_config_dict:
                _vision_config_dict["id2label"] = {
                    str(key): value for key, value in _vision_config_dict["id2label"].items()
                }

            # Give a warning if the values exist in both `_vision_config_dict` and `vision_config` but being different.
            for key, value in _vision_config_dict.items():
                if key in vision_config and value != vision_config[key] and key not in ["transformers_version"]:
                    # If specified in `vision_config_dict`
                    if key in vision_config_dict:
                        message = (
                            f"`{key}` is found in both `vision_config_dict` and `vision_config` but with different "
                            f'values. The value `vision_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`vision_config_dict` is provided which will be used to initialize `MuMURVisionConfig`. "
                            f'The value `vision_config["{key}"]` will be overriden.'
                        )
                    logger.warning(message)

            # Update all values in `vision_config` with the ones in `_vision_config_dict`.
            vision_config.update(_vision_config_dict)

        if multilingual_text_config_dict is not None:
            if multilingual_text_config is None:
                multilingual_text_config = {}

            # This is the complete result when using `text_config_dict`.
            _multilingual_text_config_dict = MuMURMultilingualTextConfig(**multilingual_text_config_dict).to_dict()

            # Give a warning if the values exist in both `_text_config_dict` and `text_config` but being different.
            for key, value in multilingual_text_config.items():
                if key in multilingual_text_config and value != multilingual_text_config[key] and key not in ["transformers_version"]:
                    # If specified in `text_config_dict`
                    if key in multilingual_text_config_dict:
                        message = (
                            f"`{key}` is found in both `multilingual_text_config_dict` and `multilingual_text_config` but with different values. "
                            f'The value `multilingual_text_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`multilingual_text_config_dict` is provided which will be used to initialize `MuMURMultilingualTextConfig`. The "
                            f'value `multilingual_text_config["{key}"]` will be overriden.'
                        )
                    logger.warning(message)

            # Update all values in `text_config` with the ones in `_multilingual_text_config_dict`.
            multilingual_text_config.update(_multilingual_text_config_dict)

        if multimodal_config_dict is not None:
            if multimodal_config is None:
                multimodal_config = {}

            # This is the complete result when using `text_config_dict`.
            _multimodal_config_dict = MuMURMultilingualTextConfig(**multimodal_config_dict).to_dict()

            # Give a warning if the values exist in both `_text_config_dict` and `text_config` but being different.
            for key, value in multimodal_config.items():
                if key in multimodal_config and value != multimodal_config[key] and key not in ["transformers_version"]:
                    # If specified in `text_config_dict`
                    if key in multimodal_config_dict:
                        message = (
                            f"`{key}` is found in both `multimodal_config_dict` and `multimodal_config` but with different values. "
                            f'The value `multimodal_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`multimodal_config_dict` is provided which will be used to initialize `MuMURMultimodalConfig`. The "
                            f'value `multimodal_config["{key}"]` will be overriden.'
                        )
                    logger.warning(message)

            # Update all values in `text_config` with the ones in `_multilingual_text_config_dict`.
            multimodal_config_dict.update(_multimodal_config_dict)

        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `MuMURTextConfig` with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. initializing the `MuMURVisionConfig` with default values.")

        if multilingual_text_config is None:
            multilingual_text_config = {}
            logger.info("`multilingual_text_config` is `None`. initializing the `MuMURMultilingualTextConfig` with default values.")

        if multimodal_config is None:
            multimodal_config = {}
            logger.info("`multimodal_config` is `None`. initializing the `MuMURMultiModalConfig` with default values.")

        self.text_config = MuMURTextConfig(**text_config)
        self.vision_config = MuMURVisionConfig(**vision_config)
        self.multilingual_text_config = MuMURMultilingualTextConfig(**multilingual_text_config)
        self.multimodal_config = MuMURMultiModalConfig(**multimodal_config)
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.extract_layers = extract_layers
        self.reduce_dim = reduce_dim
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_attention_dropout = decoder_attention_dropout
        self.decoder_hidden_act = decoder_hidden_act
        self.decoder_intermediate_size = decoder_intermediate_size
        self.conditional_layer = conditional_layer
        self.initializer_factor = 1.0
        self.use_complex_transposed_convolution = use_complex_transposed_convolution

    @classmethod
    def from_text_vision_configs(cls, text_config: MuMURTextConfig, vision_config: MuMURVisionConfig, **kwargs):
        r"""
        Instantiate a [`MuMURConfig`] (or a derived class) from mumur text model configuration and mumur vision
        model configuration.

        Returns:
            [`MuMURConfig`]: An instance of a configuration object
        """

        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)
