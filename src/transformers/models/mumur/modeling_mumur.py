# coding=utf-8
# Copyright 2023 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
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
""" PyTorch MuMUR model."""

import copy
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_mumur import MuMURConfig, MuMURTextConfig, MuMURMultilingualTextConfig, MuMURVisionConfig, MuMURMultiModalConfig
from ..auto import AutoModel

logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "Intel/mumur"

MUMUR_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Intel/mumur",
    # See all MuMUR models at https://huggingface.co/models?filter=mumur
]



# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


# Copied from transformers.models.clip.modeling_clip.clip_loss with clip->mumur
def mumur_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPOutput with CLIP->MuMUR
class MuMUROutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`MuMURTextModel`].
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            [`MuMURVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`MuMURTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`MuMURVisionModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    text_embeds: torch.FloatTensor = None
    m_text_embeds: torch.FloatTensor = None
    vision_embeds_pooled: torch.FloatTensor = None
    vision_embeds_pooled_multilingual: torch.FloatTensor = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )

@dataclass
class MuMURMutlimodalOutput(ModelOutput):
    """
    Args:  #TODO change args
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`MuMURTextModel`].
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            [`MuMURVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`MuMURTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`MuMURVisionModel`].
    """

    last_hidden_state: torch.FloatTensor = None
    attentions: torch.FloatTensor = None


    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


@dataclass
# Copied from transformers.models.clipseg.modeling_clipseg.CLIPSegDecoderOutput with CLIPSeg->MuMUR
class MuMURDecoderOutput(ModelOutput):
    """
    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, height, width)`):
            Classification scores for each pixel.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# Copied from transformers.models.clipseg.modeling_clipseg.CLIPSegVisionEmbeddings with CLIPSeg->MuMUR
class MuMURVisionEmbeddings(nn.Module):
    def __init__(self, config: MuMURVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def interpolate_position_embeddings(self, new_size):
        if len(new_size) != 2:
            raise ValueError("new_size should consist of 2 values")

        num_patches_one_direction = int(self.num_patches**0.5)
        # we interpolate the position embeddings in 2D
        a = self.position_embedding.weight[1:].T.view(
            1, self.config.hidden_size, num_patches_one_direction, num_patches_one_direction
        )
        b = (
            nn.functional.interpolate(a, new_size, mode="bicubic", align_corners=False)
            .squeeze(0)
            .view(self.config.hidden_size, new_size[0] * new_size[1])
            .T
        )
        result = torch.cat([self.position_embedding.weight[:1], b])

        return result

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        if embeddings.shape[1] != self.num_positions:
            new_shape = int(math.sqrt(embeddings.shape[1] - 1))
            embeddings = embeddings + self.interpolate_position_embeddings((new_shape, new_shape))
            embeddings = embeddings.to(embeddings.dtype)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


# Copied from transformers.models.clip.modeling_clip.CLIPTextEmbeddings with CLIP->MuMUR
class MuMURTextEmbeddings(nn.Module):
    def __init__(self, config: MuMURTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings
    
# Copied from transformers.models.clip.modeling_clip.CLIPTextEmbeddings with CLIP->MuMUR
class MuMURTextEmbeddings(nn.Module):
    def __init__(self, config: MuMURTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings



# Copied from transformers.models.clip.modeling_clip.CLIPAttention with CLIP->MuMUR
class MuMURAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped

# Copied from transformers.models.clip.modeling_clip.CLIPAttention with CLIP->MuMUR
class MuMURMultimodalAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper""" #TODOchange this

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        text_hidden_states: torch.Tensor,
        vision_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        n_text, _ = text_hidden_states.shape
        n_vision, n_frames, _ = vision_hidden_states.shape
        # get query proj
        query_states = self.q_proj(text_hidden_states) 
        query_states = query_states.reshape(n_text, self.num_heads, self.head_dim)
        query_states = query_states.permute(1,2,0)

        key_states = self.k_proj(vision_hidden_states)
        key_states = key_states.reshape(n_vision, n_frames, self.num_heads, self.head_dim)
        key_states = key_states.permute(0,2,1,3)

        value_states = self.v_proj(vision_hidden_states)
        value_states = value_states.reshape(n_vision, n_frames, self.num_heads, self.head_dim)
        value_states = value_states.permute(0,2,3,1)

        # proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        # query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        # key_states = key_states.view(*proj_shape)
        # value_states = value_states.view(*proj_shape)

        # src_len = key_states.size(1)
        attn_weights = key_states@query_states
        attn_weights = attn_weights / math.sqrt(self.head_dim)  #TODO check wherte this was in original code

        # if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        #     raise ValueError(
        #         f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
        #         f" {attn_weights.size()}"
        #     )

        # # apply the causal_attention_mask first
        # if causal_attention_mask is not None:
        #     if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
        #         raise ValueError(
        #             f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
        #             f" {causal_attention_mask.size()}"
        #         )
        #     attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
        #     attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # if attention_mask is not None:
        #     if attention_mask.size() != (bsz, 1, tgt_len, src_len):
        #         raise ValueError(
        #             f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
        #         )
        #     attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        #     attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=2)

        # if output_attentions:  #TODO
        #     # this operation is a bit akward, but it's required to
        #     # make sure that attn_weights keeps its gradient.
        #     # In order to do so, attn_weights have to reshaped
        #     # twice and have to be reused in the following
        #     attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        #     attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        # else:
        #     attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = value_states@attn_probs
        attn_output = attn_output.permute(0,3,1,2)
        attn_output = attn_output.reshape(n_vision, n_text, self.embed_dim)

        # if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        #     raise ValueError(
        #         f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
        #         f" {attn_output.size()}"
        #     )

        # attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        # attn_output = attn_output.transpose(1, 2)
        # attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)
        attn_weights_reshaped  = None #TODO change
        return attn_output, attn_weights_reshaped


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->MuMUR
class MuMURMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer with CLIP->MuMUR
class MuMUREncoderLayer(nn.Module):
    def __init__(self, config: MuMURConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = MuMURAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MuMURMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# # Copied from transformers.models.clipseg.modeling_clipseg.CLIPSegPreTrainedModel with CLIPSeg->MuMUR
class MuMURPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MuMURConfig
    base_model_prefix = "clip"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, MuMURTextEmbeddings):
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, MuMURVisionEmbeddings):
            factor = self.config.initializer_factor
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        elif isinstance(module, MuMURAttention):
            factor = self.config.initializer_factor
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, MuMURMLP):
            factor = self.config.initializer_factor
            in_proj_std = (
                (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            )
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        elif isinstance(module, MuMURModel):
            nn.init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
            )
            nn.init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
            )

        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, MuMUREncoder):
            module.gradient_checkpointing = value


MUMUR_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MuMURConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MUMUR_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

MUMUR_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

MUMUR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# Copied from transformers.models.clip.modeling_clip.CLIPEncoder with CLIP->MuMUR
class MuMUREncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`MuMUREncoderLayer`].

    Args:
        config: MuMURConfig
    """

    def __init__(self, config: MuMURConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([MuMUREncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.clipseg.modeling_clipseg.CLIPSegTextTransformer with CLIPSEG->MUMUR,CLIPSeg->MuMUR,clipseg->mumur
class MuMURTextTransformer(nn.Module):
    def __init__(self, config: MuMURTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = MuMURTextEmbeddings(config)
        self.encoder = MuMUREncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        # For `pooled_output` computation
        self.eos_token_id = config.eos_token_id

    @add_start_docstrings_to_model_forward(MUMUR_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=MuMURTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # MuMUR's text model uses causal mask, prepare it here.
        # https://github.com/openai/MuMUR/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/mumur/model.py#L324
        causal_attention_mask = _make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        if self.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A MuMUR model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# Copied from transformers.models.clipseg.modeling_clipseg.CLIPSegTextModel with CLIPSEG->MUMUR,CLIPSeg->MuMUR,CIDAS/clipseg-rd64-refined->Intel/mumur
class MuMURTextModel(nn.Module):
    config_class = MuMURTextConfig

    def __init__(self, config: MuMURTextConfig):
        super().__init__(config)
        self.text_model = MuMURTextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    @add_start_docstrings_to_model_forward(MUMUR_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=MuMURTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, MuMURTextModel

        >>> tokenizer = AutoTokenizer.from_pretrained("Intel/mumur")
        >>> model = MuMURTextModel.from_pretrained("Intel/mumur")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )



class MuMURMultilingualTextTransformer(nn.Module):
    config_class = MuMURMultilingualTextConfig


    def __init__(self, config: MuMURMultilingualTextConfig):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(config.model_name) #ToDO to copy from
        self.linear = torch.nn.Linear(in_features=config.hidden_size, out_features=config.output_dim) #TODO why /2? where does it come from

        # Initialize weights and apply final processing
        # self.post_init()

    # def get_input_embeddings(self) -> nn.Module:
    #     return self.text_model.embeddings.token_embedding

    # def set_input_embeddings(self, value):
    #     self.text_model.embeddings.token_embedding = value

    @add_start_docstrings_to_model_forward(MUMUR_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=MuMURTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, MuMURMultilingualTextModel

        >>> tokenizer = AutoTokenizer.from_pretrained("M-CLIP/XLM-Roberta-Large-Vit-B-32")
        >>> model = MuMURMultilingualTextModel.from_pretrained("M-CLIP/XLM-Roberta-Large-Vit-B-32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        embeddings = outputs.last_hidden_state

        embeddings = (embeddings * attention_mask.unsqueeze(2)).sum(dim=1) / attention_mask.sum(dim=1)[:, None]
        return self.linear(embeddings)
    
    @classmethod
    def _load_state_dict_into_model(cls, model, state_dict, pretrained_model_name_or_path, _fast_init=True):
        model.load_state_dict(state_dict)
        return model, [], [], []

# Copied from transformers.models.clipseg.modeling_clipseg.CLIPSegTextModel with CLIPSEG->MUMUR,CLIPSeg->MuMUR,CIDAS/clipseg-rd64-refined->Intel/mumur
class MuMURMultilingualTextModel(nn.Module):
    config_class = MuMURMultilingualTextConfig

    def __init__(self, config: MuMURMultilingualTextConfig):
        super().__init__(config)
        self.m_text_model = MuMURMultilingualTextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.m_text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.m_text_model.embeddings.token_embedding = value

    @add_start_docstrings_to_model_forward(MUMUR_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=MuMURTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, MuMURTextModel

        >>> tokenizer = AutoTokenizer.from_pretrained("Intel/mumur")
        >>> model = MuMURTextModel.from_pretrained("Intel/mumur")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        return self.m_text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# Copied from transformers.models.clipseg.modeling_clipseg.CLIPSegVisionTransformer with CLIPSEG->MUMUR,CLIPSeg->MuMUR
class MuMURVisionTransformer(nn.Module):
    def __init__(self, config: MuMURVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = MuMURVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = MuMUREncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    @add_start_docstrings_to_model_forward(MUMUR_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=MuMURVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# Copied from transformers.models.clipseg.modeling_clipseg.CLIPSegVisionModel with CLIPSEG->MUMUR,CLIPSeg->MuMUR,CIDAS/clipseg-rd64-refined->Intel/mumur
class MuMURVisionModel(MuMURPreTrainedModel):
    config_class = MuMURVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: MuMURVisionConfig):
        super().__init__(config)
        self.vision_model = MuMURVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(MUMUR_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=MuMURVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, MuMURVisionModel

        >>> processor = AutoProcessor.from_pretrained("Intel/mumur")
        >>> model = MuMURVisionModel.from_pretrained("Intel/mumur")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

class MuMURMutlimodalTransformer(nn.Module):
    def __init__(self, config: MuMURMultiModalConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.output_dim = config.output_dim
        self.attention_dropout = config.attention_dropout
        self.cross_attn = MuMURMultimodalAttention(config)

        self.linear_proj = nn.Linear(self.output_dim, self.output_dim) #TODO check size why not hidden sytate
        self.init_layer_norm = nn.LayerNorm(self.output_dim)
        self.mid_layer_norm = nn.LayerNorm(self.output_dim)
        self.last_layer_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(config.attention_dropout)

        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    # @add_start_docstrings_to_model_forward(MUMUR_VISION_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=MuMURVisionConfig)
    # def forward(
    #     self,
    #     pixel_values: Optional[torch.FloatTensor] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    # ) -> Union[Tuple, BaseModelOutputWithPooling]:
    #     r"""
    #     Returns:

    #     """
    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    #     if pixel_values is None:
    #         raise ValueError("You have to specify pixel_values")

    #     hidden_states = self.embeddings(pixel_values)
    #     hidden_states = self.pre_layrnorm(hidden_states)

    #     encoder_outputs = self.encoder(
    #         inputs_embeds=hidden_states,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )

    #     last_hidden_state = encoder_outputs[0]
    #     pooled_output = last_hidden_state[:, 0, :]
    #     pooled_output = self.post_layernorm(pooled_output)

    #     if not return_dict:
    #         return (last_hidden_state, pooled_output) + encoder_outputs[1:]

    def forward(self, 
                text_embeds: Optional[torch.FloatTensor] = None,
                video_embeds: Optional[torch.FloatTensor] = None
                )-> Union[Tuple, BaseModelOutputWithPooling]:
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            out: num_vids x num_texts x embed_dim
        """
        text_embeds = self.init_layer_norm(text_embeds)
        video_embeds = self.init_layer_norm(video_embeds)

        # num_vids x num_texts x embed_dim
        mm_attention, _ = self.cross_attn(text_embeds, video_embeds)
        mm_attention = self.mid_layer_norm(mm_attention)

        linear_out = self.linear_proj(mm_attention)
        mm_output = self.last_layer_norm(linear_out)

        return MuMURMutlimodalOutput(
            last_hidden_state=mm_output, 
            attentions=mm_attention, 
        )

@add_start_docstrings(MUMUR_START_DOCSTRING)
# Copied from transformers.models.clipseg.modeling_clipseg.CLIPSegModel with CLIPSEG->MUMUR,CLIPSeg->MuMUR,clipseg->mumur,CIDAS/clipseg-rd64-refined->Intel/mumur
class MuMURModel(MuMURPreTrainedModel):
    config_class = MuMURConfig

    def __init__(self, config: MuMURConfig):
        super().__init__(config)

        if not isinstance(config.text_config, MuMURTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type MuMURTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, MuMURVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type MuMURVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config
        multilingual_text_config = config.multilingual_text_config
        multimodal_config = config.multimodal_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.m_text_embed_dim = multilingual_text_config.output_dim
        self.visual_embed_dim = vision_config.hidden_size

        self.vision_embed_dim = vision_config.hidden_size
        self.num_frames = vision_config.num_frames

        self.text_model = MuMURTextTransformer(text_config)
        self.mulitlingual_text_model = MuMURMultilingualTextTransformer(multilingual_text_config)
        self.vision_model = MuMURVisionTransformer(vision_config)

        self.multimodal_projection = MuMURMutlimodalTransformer(multimodal_config)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.m_text_projection = nn.Linear(self.m_text_embed_dim, self.projection_dim, bias=False)
        self.visual_projection = nn.Linear(self.visual_embed_dim, self.projection_dim, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MUMUR_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`MuMURTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, MuMURModel

        >>> tokenizer = AutoTokenizer.from_pretrained("Intel/mumur")
        >>> model = MuMURModel.from_pretrained("Intel/mumur")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use MUMUR model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)

        return text_features


    @add_start_docstrings_to_model_forward(MUMUR_TEXT_INPUTS_DOCSTRING)
    def get_multiling_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`MuMURTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, MuMURModel

        >>> tokenizer = AutoTokenizer.from_pretrained("Intel/mumur")
        >>> model = MuMURModel.from_pretrained("Intel/mumur")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use MUMUR model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        m_text_outputs = self.mulitlingual_text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = m_text_outputs[1]
        multiling_text_features = self.m_text_projection(pooled_output)

        return multiling_text_features

    @add_start_docstrings_to_model_forward(MUMUR_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`MuMURVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, MuMURModel

        >>> processor = AutoProcessor.from_pretrained("Intel/mumur")
        >>> model = MuMURModel.from_pretrained("Intel/mumur")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # Use MUMUR model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features

    @add_start_docstrings_to_model_forward(MUMUR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MuMUROutput, config_class=MuMURConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MuMUROutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, MuMURModel

        >>> processor = AutoProcessor.from_pretrained("Intel/mumur")
        >>> model = MuMURModel.from_pretrained("Intel/mumur")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        # Use MUMUR model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        m_text_outputs = self.mulitlingual_text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        batch_size = input_ids.shape[0] #TODO
        vision_embeds = vision_outputs[1]
        vision_embeds = self.visual_projection(vision_embeds)
        vision_embeds = vision_embeds.reshape(batch_size, self.num_frames, -1)


        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        m_text_embeds = m_text_outputs
        m_text_embeds = self.m_text_projection(m_text_embeds)



        vision_embeds_pooled = self.multimodal_projection(text_embeds, vision_embeds).last_hidden_state
        vision_embeds_pooled_multilingual = self.multimodal_projection(m_text_embeds, vision_embeds).last_hidden_state
     
        # normalized features
        vision_embeds_pooled = vision_embeds_pooled / vision_embeds_pooled.norm(p=2, dim=-1, keepdim=True) #TODO should we?
        vision_embeds_pooled_multilingual = vision_embeds_pooled_multilingual / vision_embeds_pooled_multilingual.norm(p=2, dim=-1, keepdim=True) #TODO should we?
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        # logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        # logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = mumur_loss(None) #TODO 

        # if not return_dict:
        #     output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
        #     return ((loss,) + output) if loss is not None else output

        return MuMUROutput(
            loss=loss,
            text_embeds=text_embeds,
            m_text_embeds=m_text_embeds,
            vision_embeds_pooled_multilingual=vision_embeds_pooled_multilingual,
            vision_embeds_pooled=vision_embeds_pooled
        )


# # Copied from transformers.models.clipseg.modeling_clipseg.CLIPSegDecoderLayer with CLIPSeg->MuMUR
# class MuMURDecoderLayer(nn.Module):
#     """
#     MuMUR decoder layer, which is identical to `MuMUREncoderLayer`, except that normalization is applied after
#     self-attention/MLP, rather than before.
#     """

#     def __init__(self, config: MuMURConfig):
#         super().__init__()
#         self.embed_dim = config.hidden_size
#         self.self_attn = MuMURAttention(config)
#         self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
#         self.mlp = MuMURMLP(config)
#         self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: torch.Tensor,
#         causal_attention_mask: torch.Tensor,
#         output_attentions: Optional[bool] = False,
#     ) -> Tuple[torch.FloatTensor]:
#         """
#         Args:
#             hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
#             attention_mask (`torch.FloatTensor`): attention mask of size
#                 `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
#                 `(config.encoder_attention_heads,)`.
#             output_attentions (`bool`, *optional*):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more detail.
#         """
#         residual = hidden_states

#         hidden_states, attn_weights = self.self_attn(
#             hidden_states=hidden_states,
#             attention_mask=attention_mask,
#             causal_attention_mask=causal_attention_mask,
#             output_attentions=output_attentions,
#         )

#         hidden_states = residual + hidden_states
#         hidden_states = self.layer_norm1(hidden_states)

#         residual = hidden_states
#         hidden_states = self.mlp(hidden_states)
#         hidden_states = residual + hidden_states
#         hidden_states = self.layer_norm2(hidden_states)

#         outputs = (hidden_states,)

#         if output_attentions:
#             outputs += (attn_weights,)

#         return outputs


# # Copied from transformers.models.clipseg.modeling_clipseg.CLIPSegDecoder with CLIPSeg->MuMUR
# class MuMURDecoder(MuMURPreTrainedModel):
#     def __init__(self, config: MuMURConfig):
#         super().__init__(config)

#         self.conditional_layer = config.conditional_layer

#         self.film_mul = nn.Linear(config.projection_dim, config.reduce_dim)
#         self.film_add = nn.Linear(config.projection_dim, config.reduce_dim)

#         if config.use_complex_transposed_convolution:
#             transposed_kernels = (config.vision_config.patch_size // 4, config.vision_config.patch_size // 4)

#             self.transposed_convolution = nn.Sequential(
#                 nn.Conv2d(config.reduce_dim, config.reduce_dim, kernel_size=3, padding=1),
#                 nn.ReLU(),
#                 nn.ConvTranspose2d(
#                     config.reduce_dim,
#                     config.reduce_dim // 2,
#                     kernel_size=transposed_kernels[0],
#                     stride=transposed_kernels[0],
#                 ),
#                 nn.ReLU(),
#                 nn.ConvTranspose2d(
#                     config.reduce_dim // 2, 1, kernel_size=transposed_kernels[1], stride=transposed_kernels[1]
#                 ),
#             )
#         else:
#             self.transposed_convolution = nn.ConvTranspose2d(
#                 config.reduce_dim, 1, config.vision_config.patch_size, stride=config.vision_config.patch_size
#             )

#         depth = len(config.extract_layers)
#         self.reduces = nn.ModuleList(
#             [nn.Linear(config.vision_config.hidden_size, config.reduce_dim) for _ in range(depth)]
#         )

#         decoder_config = copy.deepcopy(config.vision_config)
#         decoder_config.hidden_size = config.reduce_dim
#         decoder_config.num_attention_heads = config.decoder_num_attention_heads
#         decoder_config.intermediate_size = config.decoder_intermediate_size
#         decoder_config.hidden_act = "relu"
#         self.layers = nn.ModuleList([MuMURDecoderLayer(decoder_config) for _ in range(len(config.extract_layers))])

#     def forward(
#         self,
#         hidden_states: Tuple[torch.Tensor],
#         conditional_embeddings: torch.Tensor,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = True,
#     ):
#         all_hidden_states = () if output_hidden_states else None
#         all_attentions = () if output_attentions else None

#         activations = hidden_states[::-1]

#         output = None
#         for i, (activation, layer, reduce) in enumerate(zip(activations, self.layers, self.reduces)):
#             if output is not None:
#                 output = reduce(activation) + output
#             else:
#                 output = reduce(activation)

#             if i == self.conditional_layer:
#                 output = self.film_mul(conditional_embeddings) * output.permute(1, 0, 2) + self.film_add(
#                     conditional_embeddings
#                 )
#                 output = output.permute(1, 0, 2)

#             layer_outputs = layer(
#                 output, attention_mask=None, causal_attention_mask=None, output_attentions=output_attentions
#             )

#             output = layer_outputs[0]

#             if output_hidden_states:
#                 all_hidden_states += (output,)

#             if output_attentions:
#                 all_attentions += (layer_outputs[1],)

#         output = output[:, 1:, :].permute(0, 2, 1)  # remove cls token and reshape to [batch_size, reduce_dim, seq_len]

#         size = int(math.sqrt(output.shape[2]))

#         batch_size = conditional_embeddings.shape[0]
#         output = output.view(batch_size, output.shape[1], size, size)

#         logits = self.transposed_convolution(output).squeeze()

#         if not return_dict:
#             return tuple(v for v in [logits, all_hidden_states, all_attentions] if v is not None)

#         return MuMURDecoderOutput(
#             logits=logits,
#             hidden_states=all_hidden_states,
#             attentions=all_attentions,
#         )


# @add_start_docstrings(
#     """
#     MuMUR model with a Transformer-based decoder on top for zero-shot and one-shot image segmentation.
#     """,
#     MUMUR_START_DOCSTRING,
# )
