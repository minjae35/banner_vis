# v2: v1에서 seg 모듈 제외

import os
import glob
from pathlib import Path

import time
from datetime import datetime

import json
import yaml
import argparse
from easydict import EasyDict

import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import Tensor
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules import transformer
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils.rnn import pad_sequence

import torchvision
from torchvision import transforms, models
from torchvision.models import resnet
from torchvision.transforms import Resize
from torchvision.transforms.functional import resize, to_pil_image 

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import lightning as L
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW

import segmentation_models_pytorch as smp
from timm.optim import create_optimizer_v2
from timm.models.helpers import named_apply
from timm.models.vision_transformer import VisionTransformer, PatchEmbed

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from PIL import Image, ImageFont
from bounding_box import bounding_box as bb

import logging
import warnings

import re
import gc
import math
import MeCab
import string
from copy import deepcopy
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import partial
from itertools import permutations
from typing import Sequence, Optional, Tuple, List, Any, Dict, Type
from collections import defaultdict, deque, OrderedDict, namedtuple

from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
from mmcv.transforms import Compose as compose_cv

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
warnings.filterwarnings('ignore')
if torch.cuda.is_available(): cudnn.benchmark = True


class CharsetAdapter:
    """Transforms labels according to the target charset."""

    def __init__(self, target_charset) -> None:
        super().__init__()
        self.lowercase_only = target_charset == target_charset.lower()
        self.uppercase_only = target_charset == target_charset.upper()
        self.unsupported = re.compile(f'[^{re.escape(target_charset)}]')

    def __call__(self, label):
        if self.lowercase_only:
            label = label.lower()
        elif self.uppercase_only:
            label = label.upper()

        #Leehakho
        #print(label)
        label = self.unsupported.sub('', label)
        #print(label)
        return label

class BaseTokenizer(ABC):

    def __init__(self, charset: str, specials_first: tuple = (), specials_last: tuple = ()) -> None:
        self._itos = specials_first + tuple(charset) + specials_last
        self._stoi = {s: i for i, s in enumerate(self._itos)}

    def __len__(self):
        return len(self._itos)

    def _tok2ids(self, tokens: str) -> List[int]:
        return [self._stoi[s] for s in tokens]

    def _ids2tok(self, token_ids: List[int], join: bool = True) -> str:
        tokens = [self._itos[i] for i in token_ids]
        return ''.join(tokens) if join else tokens

    @abstractmethod
    def encode(self, labels: List[str], device: Optional[torch.device] = None) -> Tensor:
        """Encode a batch of labels to a representation suitable for the model.

        Args:
            labels: List of labels. Each can be of arbitrary length.
            device: Create tensor on this device.

        Returns:
            Batched tensor representation padded to the max label length. Shape: N, L
        """
        raise NotImplementedError

    @abstractmethod
    def _filter(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        """Internal method which performs the necessary filtering prior to decoding."""
        raise NotImplementedError

    def decode(self, token_dists: Tensor, raw: bool = False) -> Tuple[List[str], List[Tensor]]:
        """Decode a batch of token distributions.

        Args:
            token_dists: softmax probabilities over the token distribution. Shape: N, L, C
            raw: return unprocessed labels (will return list of list of strings)

        Returns:
            list of string labels (arbitrary length) and
            their corresponding sequence probabilities as a list of Tensors
        """
        batch_tokens = []
        batch_probs = []
        for dist in token_dists:
            probs, ids = dist.max(-1)  # greedy selection
            if not raw:
                probs, ids = self._filter(probs, ids)
            tokens = self._ids2tok(ids, not raw)
            batch_tokens.append(tokens)
            batch_probs.append(probs)
        return batch_tokens, batch_probs

class Tokenizer(BaseTokenizer):
    BOS = '[B]'
    EOS = '[E]'
    PAD = '[P]'

    def __init__(self, charset: str) -> None:
        specials_first = (self.EOS,)
        specials_last = (self.BOS, self.PAD)
        super().__init__(charset, specials_first, specials_last)
        self.eos_id, self.bos_id, self.pad_id = [self._stoi[s] for s in specials_first + specials_last]

    def encode(self, labels: List[str], device: Optional[torch.device] = None) -> Tensor:
        batch = [torch.as_tensor([self.bos_id] + self._tok2ids(y) + [self.eos_id], dtype=torch.long, device=device)
                 for y in labels]
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)

    def _filter(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        ids = ids.tolist()
        try:
            eos_idx = ids.index(self.eos_id)
        except ValueError:
            eos_idx = len(ids)  # Nothing to truncate.
        # Truncate after EOS
        ids = ids[:eos_idx]
        probs = probs[:eos_idx + 1]  # but include prob. for EOS (if it exists)
        return probs, ids

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        ## 기존에는 self.iter = 3회만큼 iteration을 두고 evolution을 했는데, 여기는 1회만 하고 끝낸다. 
        self.evolve_gcn = Transformer(256, 128, num_heads=8,
                                        dim_feedforward=1024, drop_rate=0.0, if_resi=True, block_nums=3)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        jitter_poly_torch: torch.Tensor, ## For poloygon prediction
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred, pred_poly = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            jitter_poly_torch=jitter_poly_torch,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        # Prepare output
        return masks, iou_pred, pred_poly

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        jitter_poly_torch: torch.Tensor, ## For poloygon prediction
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        # b, c, h, w = upscaled_embedding.shape  ## 135번째 라인의 shape와 중복되지 않도록 수정하기. 
        b_up, c_up, h_up, w_up = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b_up, c_up, h_up * w_up)).view(b_up, -1, h_up, w_up)


        # print(self.output_hypernetworks_mlps[0].layers[0].weight)
        # print(self.output_hypernetworks_mlps[0].layers[0].weight)
        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        #### point prediction
        # img_poly =  torch.zeros((b, 64, 2)).to(src.device)   ## 여기서 이니셜을 0으로 하는 부분을 없애야함. 
        # print("jiter_poly_torch", jitter_poly_torch)
        img_poly = jitter_poly_torch.clone().to(src.device)
        # ind = torch.from_numpy(np.array([0])).to(src.device) ## 이 부분은 batch size가 1이라고 가정하고 만든 부분이라서, 디버깅 과정중이라 항상 batch가 1로 들어오나? 확인 필요
        ind = torch.arange(b).to(src.device)
        node_feature = get_node_feature(src, img_poly=img_poly, ind=ind, h=h, w=w)  ## 앞선 150번째 라인때문에 node feature가 제대로 안뽑히는 결과가 나옴. 
        evolve_gcn_transposed = self.evolve_gcn(node_feature).transpose(1, 2)

        # print("evolve_gcn_transposed", evolve_gcn_transposed)
        i_poly = img_poly + evolve_gcn_transposed
        pred_poly = torch.clamp(i_poly, 0, w-1)


        return masks, iou_pred, pred_poly
    
def get_node_feature(cnn_feature, img_poly, ind, h, w):
    img_poly = img_poly.clone().float()
    img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
    img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1

    batch_size = cnn_feature.size(0)
    gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(img_poly.device)

    for i in range(batch_size):
        poly = img_poly[ind == i].unsqueeze(0)
        results = torch.nn.functional.grid_sample(cnn_feature[i:i + 1], poly, align_corners=False)[0].permute(1, 0, 2)
        gcn_feature[ind == i] = results
    return gcn_feature

class Transformer(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads=8,
                 dim_feedforward=1024, drop_rate=0.1, if_resi=False, block_nums=3):
        super().__init__()

        self.bn0 = nn.BatchNorm1d(in_dim, affine=False)
        self.conv1 = nn.Conv1d(in_dim, out_dim, 1, dilation=1)

        # self.pos_embedding = Positional_encoding(in_dim)

                
        self.transformer = TransformerLayer(out_dim, in_dim, num_heads, attention_size=out_dim,
                                            dim_feedforward=dim_feedforward, drop_rate=drop_rate,
                                            if_resi=if_resi, block_nums=block_nums)

        self.prediction = nn.Sequential(
            nn.Conv1d(2*out_dim, 128, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
            nn.Conv1d(64, 2, 1))

    def forward(self, x):
        x = self.bn0(x)
        x1 = x.permute(0, 2, 1)
        x1 = self.transformer(x1)
        x1 = x1.permute(0, 2, 1)

        x = torch.cat([x1, self.conv1(x)], dim=1)
        pred = self.prediction(x)

        return pred
    
class TransformerLayer(nn.Module):
    def __init__(self, out_dim, in_dim, num_heads, attention_size,
                 dim_feedforward=1024, drop_rate=0.1, if_resi=True, block_nums=3):
        super(TransformerLayer, self).__init__()
        self.block_nums = block_nums
        self.if_resi = if_resi
        self.linear = nn.Linear(in_dim, attention_size)
        for i in range(self.block_nums):
            self.__setattr__('MHA_self_%d' % i, MultiHeadAttention(num_heads, attention_size,
                                                                   dropout=drop_rate, if_resi=if_resi))
            self.__setattr__('FFN_%d' % i, FeedForward(out_dim, dim_feedforward, if_resi=if_resi))

    def forward(self, query):
        inputs = self.linear(query)
        # outputs = inputs
        for i in range(self.block_nums):
            outputs = self.__getattr__('MHA_self_%d' % i)(inputs)
            outputs = self.__getattr__('FFN_%d' % i)(outputs)
            if self.if_resi:
                inputs = inputs+outputs
            else:
                inputs = outputs
        # outputs = inputs
        return inputs

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout=0.1, if_resi=True):
        super(MultiHeadAttention, self).__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.MultiheadAttention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.Q_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())
        self.if_resi = if_resi

    def forward(self, inputs):
        query = self.layer_norm(inputs)
        q = self.Q_proj(query)
        k = self.K_proj(query)
        v = self.V_proj(query)
        attn_output, attn_output_weights = self.MultiheadAttention(q, k, v)
        if self.if_resi:
            attn_output += inputs
        else:
            attn_output = attn_output

        return attn_output

class FeedForward(nn.Module):
    def __init__(self, in_channel, FFN_channel, if_resi=True):
        super(FeedForward, self).__init__()
        """
        1024 2048
        """
        output_channel = (FFN_channel, in_channel)
        self.fc1 = nn.Sequential(nn.Linear(in_channel, output_channel[0]), nn.ReLU())
        self.fc2 = nn.Linear(output_channel[0], output_channel[1])
        self.layer_norm = nn.LayerNorm(in_channel)
        self.if_resi = if_resi

    def forward(self, inputs):
        outputs = self.layer_norm(inputs)
        outputs = self.fc1(outputs)
        outputs = self.fc2(outputs)
        if self.if_resi:
            outputs += inputs
        else:
            outputs = outputs
        return outputs  

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        # print(self.layers[0].weight)
        return x

class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        name: str,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        self.name = name
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # feature = keys.permute(0, 2, 1).view(1, 256, 64, 64)
        # for a in range(256):
        # save_plt(feature[0, a], a)
            
        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        feature = keys.permute(0, 2, 1).view(1, 256, 64, 64)
        
        # for a in range(256):
        #     save_plt(feature[0, a], 'att_a'+ str(a))
            

        return queries, keys

class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
          
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # print(queries.shape, query_pe.shape, keys.shape, key_pe.shape)
        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        

        return queries, keys

class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim)
                            for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (
            4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans,
                      kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

        # promt finetuning
        mlp_dim = 2048
        num_heads = 8
        attention_downsample_rate = 2
        self.modulate_prompt = TwoWayAttentionBlock(
            embedding_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            activation=activation,
            attention_downsample_rate=attention_downsample_rate,
            skip_first_layer_pe=False,
        )

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        feature: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros(
                (points.shape[0], 1, 2), device=points.device)
            padding_label = - \
                torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size)

        # point point_embedding = (B, N, 256)
        # Embedded feature (B, 256, 64, 64)
        # 256 64 64
        image_embedding = feature.flatten(1).unsqueeze(0).permute(0, 2, 1)
        # 1 4096 256
        # 1 4096 256
        image_pe = self.get_dense_pe().flatten(2).permute(0, 2, 1)

        queries = point_embedding
        keys = image_embedding

        # point promt finetuning
        queries, keys = self.modulate_prompt(
            queries=queries,
            keys=keys,
            query_pe=point_embedding,
            key_pe=image_pe,
        )
        point_embedding = point_embedding + queries

        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(
            coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        feature: Optional[torch.Tensor],
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(
                coords, labels, feature, pad=(boxes is None))
            # point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat(
                [sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat(
                [sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -
                1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        # 1 , 2,  256

        return sparse_embeddings, dense_embeddings

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape

        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        results = torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

        return results

    def re_pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape

        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        results = torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

        return results

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

def save_plt(image, path=None):
    if type(image) is torch.Tensor:
        image = image.detach().cpu().numpy()
    save_base = os.path.join('./feature_results')
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    filename = f"{path}.png"
    plt.savefig(os.path.join(save_base, filename),
                bbox_inches='tight', pad_inches=0)
    plt.close()

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (
                window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(
            dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

class Attention_(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(
                torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(
                torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(
            B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -
                            1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x

def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size,
               Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous(
    ).view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)

def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(
            -1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + \
        (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]

def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) +
        rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn

class PatchEmbed_(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x

class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed_(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size,
                            img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x

class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """         
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False) 
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

def build_sam_vit_h(checkpoint=None, device="cuda"):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=(7, 15, 23, 31),
        checkpoint=checkpoint,
        device=device
    )

build_sam = build_sam_vit_h

def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=(5, 11, 17, 23),
        checkpoint=checkpoint,
    )

def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=(2, 5, 8, 11),
        checkpoint=checkpoint,
    )

sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}

def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    device="cuda",
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
                name='mask_decoder'
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            loaded_checkpoint = torch.load(f, map_location=device)

            # If 'state_dict' key exists in the checkpoint, use it.

            ## 기학습 모델을 사용할 떄,
            if 'state_dict' in loaded_checkpoint:
                state_dict = loaded_checkpoint['state_dict']
                # Modify the keys by removing the "model." prefix
                new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
                sam.load_state_dict(new_state_dict, strict=False)

            ## SAM pth 모델을 사용할 때.
            else:
                state_dict = loaded_checkpoint
                sam.load_state_dict(state_dict, strict=False)
    return sam

class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_coords_torch(self, coords: torch.tensor, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        # coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords
    
    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    # ch_in, ch_out, kernel, stride, padding, groups
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p),
                              groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

def attempt_download(weights):
    # Attempt to download pretrained weights if not found locally
    weights = str(weights).strip().replace("'", '')
    file = Path(weights).name.lower()

    msg = weights + ' missing, try downloading from https://github.com/ultralytics/yolov5/releases/'
    models = ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']  # available models
    redundant = False  # offer second download option

    if file in models and not os.path.isfile(weights):
        # Google Drive
        # d = {'yolov5s.pt': '1R5T6rIyy3lLwgFXNms8whc-387H0tMQO',
        #      'yolov5m.pt': '1vobuEExpWQVpXExsJ2w-Mbf3HJjWkQJr',
        #      'yolov5l.pt': '1hrlqD1Wdei7UT4OgT785BEk1JwnSvNEV',
        #      'yolov5x.pt': '1mM8aZJlWTxOg7BZJvNUMrTnA2AbeCVzS'}
        # r = gdrive_download(id=d[file], name=weights) if file in d else 1
        # if r == 0 and os.path.exists(weights) and os.path.getsize(weights) > 1E6:  # check
        #    return

        try:  # GitHub
            url = 'https://github.com/ultralytics/yolov5/releases/download/v3.1/' + file
            print('Downloading %s to %s...' % (url, weights))
            torch.hub.download_url_to_file(url, weights)
            assert os.path.exists(weights) and os.path.getsize(weights) > 1E6  # check
        except Exception as e:  # GCP
            print('Download error: %s' % e)
            assert redundant, 'No secondary mirror'
            url = 'https://storage.googleapis.com/ultralytics/yolov5/ckpt/' + file
            print('Downloading %s to %s...' % (url, weights))
            r = os.system('curl -L %s -o %s' % (url, weights))  # torch.hub.download_url_to_file(url, weights)
        finally:
            if not (os.path.exists(weights) and os.path.getsize(weights) > 1E6):  # check
                os.remove(weights) if os.path.exists(weights) else None  # remove partial downloads
                print('ERROR: Download failure: %s' % msg)
            print('')
            return

class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.cat(y, 1)  # nms ensemble
        y = torch.stack(y).mean(0)  # mean ensemble
        return y, None  # inference, train output

def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        model.append(torch.load(w, map_location=map_location)[
                     'model'].float().fuse().eval())  # load FP32 model
        
    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        
    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble

@dataclass
class BatchResult:
    num_samples: int
    correct: int
    ned: float
    confidence: float
    label_length: int
    loss: Tensor
    loss_numel: int

class BaseSystem(pl.LightningModule, ABC):

    def __init__(self, tokenizer: BaseTokenizer, charset_test: str,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.charset_adapter = CharsetAdapter(charset_test)
        self.batch_size = batch_size
        self.lr = lr
        self.warmup_pct = warmup_pct
        self.weight_decay = weight_decay

    @abstractmethod
    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        """Inference

        Args:
            images: Batch of images. Shape: N, Ch, H, W
            max_length: Max sequence length of the output. If None, will use default.

        Returns:
            logits: N, L, C (L = sequence length, C = number of classes, typically len(charset_train) + num specials)
        """
        raise NotImplementedError

    @abstractmethod
    def forward_logits_loss(self, images: Tensor, labels: List[str]) -> Tuple[Tensor, Tensor, int]:
        """Like forward(), but also computes the loss (calls forward() internally).

        Args:
            images: Batch of images. Shape: N, Ch, H, W
            labels: Text labels of the images

        Returns:
            logits: N, L, C (L = sequence length, C = number of classes, typically len(charset_train) + num specials)
            loss: mean loss for the batch
            loss_numel: number of elements the loss was calculated from
        """
        raise NotImplementedError

    def configure_optimizers(self):
        agb = self.trainer.accumulate_grad_batches
        # Linear scaling so that the effective learning rate is constant regardless of the number of GPUs used with DDP.
        lr_scale = agb * \
            math.sqrt(self.trainer.num_devices) * self.batch_size / 256.
        lr = lr_scale * self.lr
        optim = create_optimizer_v2(self, 'adamw', lr, self.weight_decay)
        sched = OneCycleLR(optim, lr, self.trainer.estimated_stepping_batches, pct_start=self.warmup_pct,
                           cycle_momentum=False)
        return {'optimizer': optim, 'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int):
        optimizer.zero_grad(set_to_none=True)

    def _eval_step(self, batch, validation: bool):
        images, labels = batch
        i = 0
        correct = 0
        total = 0
        ned = 0
        confidence = 0
        label_length = 0
        if validation:
            logits, loss, loss_numel = self.forward_logits_loss(images, labels)
        else:
            # At test-time, we shouldn't specify a max_label_length because the test-time charset used
            # might be different from the train-time charset. max_label_length in eval_logits_loss() is computed
            # based on the transformed label, which could be wrong if the actual gt label contains characters existing
            # in the train-time charset but not in the test-time charset. For example, "aishahaleyes.blogspot.com"
            # is exactly 25 characters, but if processed by CharsetAdapter for the 36-char set, it becomes 23 characters
            # long only, which sets max_label_length = 23. This will cause the model prediction to be truncated.
            logits = self.forward(images)
            # Only used for validation; not needed at test-time.
            loss = loss_numel = None

        probs = logits.softmax(-1)
        preds, probs = self.tokenizer.decode(probs)
        return preds, probs

    @staticmethod
    def _aggregate_results(outputs) -> Tuple[float, float, float]:
        if not outputs:
            return 0., 0., 0.
        total_loss = 0
        total_loss_numel = 0
        total_n_correct = 0
        total_norm_ED = 0
        total_size = 0
        # Leehakho
        # total_size += 0.00000001
        for result in outputs:
            result = result['output']
            total_loss += result.loss_numel * result.loss
            total_loss_numel += result.loss_numel
            total_n_correct += result.correct
            total_norm_ED += result.ned
            total_size += result.num_samples
        # print(total_n_correct, total_size)
        acc = total_n_correct / total_size
        ned = (1 - total_norm_ED / total_size)
        loss = total_loss / total_loss_numel
        return acc, ned, loss

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, True)

    def validation_epoch_end(self, outputs) -> None:
        acc, ned, loss = self._aggregate_results(outputs)
        self.log('val_accuracy', 100 * acc, sync_dist=True)
        self.log('val_NED', 100 * ned, sync_dist=True)
        self.log('val_loss', loss, sync_dist=True)
        self.log('hp_metric', acc, sync_dist=True)

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, False)

def init_weights_(module: nn.Module, name: str = '', exclude: Sequence[str] = ()):
    """Initialize the weights using the typical initialization schemes used in SOTA models."""
    if any(map(name.startswith, exclude)):
        return
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(
            module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

class CrossEntropySystem(BaseSystem):

    def __init__(self, charset_train: str, charset_test: str,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float) -> None:
        tokenizer = Tokenizer(charset_train)
        super().__init__(tokenizer, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.pad_id = tokenizer.pad_id

    def forward_logits_loss(self, images: Tensor, labels: List[str]) -> Tuple[Tensor, Tensor, int]:
        targets = self.tokenizer.encode(labels, self.device)
        targets = targets[:, 1:]  # Discard <bos>
        max_len = targets.shape[1] - 1  # exclude <eos> from count
        logits = self.forward(images, max_len)
        print(logits.shape, targets.shape)
        loss = F.cross_entropy(logits.flatten(
            end_dim=1), targets.flatten(), ignore_index=self.pad_id)
        loss_numel = (targets != self.pad_id).sum()
        return logits, loss, loss_numel
    
class DecoderLayer(nn.Module):
    """A Transformer decoder layer supporting two-stream attention (XLNet)
       This implements a pre-LN decoder, as opposed to the post-LN default in PyTorch."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu',
                 layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_q = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_c = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = transformer._get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super().__setstate__(state)

    def forward_stream(self, tgt: Tensor, tgt_norm: Tensor, tgt_kv: Tensor, memory: Tensor, tgt_mask: Optional[Tensor],
                       tgt_key_padding_mask: Optional[Tensor]):
        """Forward pass for a single stream (i.e. content or query)
        tgt_norm is just a LayerNorm'd tgt. Added as a separate parameter for efficiency.
        Both tgt_kv and memory are expected to be LayerNorm'd too.
        memory is LayerNorm'd by ViT.
        """
        tgt2, sa_weights = self.self_attn(tgt_norm, tgt_kv, tgt_kv, attn_mask=tgt_mask,
                                          key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)

        tgt2, ca_weights = self.cross_attn(self.norm1(tgt), memory, memory)
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(tgt)))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, sa_weights, ca_weights

    def forward(self, query, content, memory, query_mask: Optional[Tensor] = None, content_mask: Optional[Tensor] = None,
                content_key_padding_mask: Optional[Tensor] = None, update_content: bool = True):
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query = self.forward_stream(query, query_norm, content_norm, memory, query_mask, content_key_padding_mask)[0]
        if update_content:
            content = self.forward_stream(content, content_norm, content_norm, memory, content_mask,
                                          content_key_padding_mask)[0]
        return query, content

class Decoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm):
        super().__init__()
        self.layers = transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, content, memory, query_mask: Optional[Tensor] = None, content_mask: Optional[Tensor] = None,
                content_key_padding_mask: Optional[Tensor] = None):
        for i, mod in enumerate(self.layers):
            last = i == len(self.layers) - 1
            query, content = mod(query, content, memory, query_mask, content_mask, content_key_padding_mask,
                                 update_content=not last)
        query = self.norm(query)
        return query

class Encoder(VisionTransformer):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed):
        super().__init__(img_size, patch_size, in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                         num_classes=0, global_pool='', class_token=False)  # these disable the classifier head

    def forward(self, x):
        # Return all tokens
        return self.forward_features(x)

class TokenEmbedding(nn.Module):

    def __init__(self, charset_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)

class PARSeq(CrossEntropySystem):

    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], patch_size: Sequence[int], embed_dim: int,
                 enc_num_heads: int, enc_mlp_ratio: int, enc_depth: int,
                 dec_num_heads: int, dec_mlp_ratio: int, dec_depth: int,
                 perm_num: int, perm_forward: bool, perm_mirrored: bool,
                 decode_ar: bool, refine_iters: int, dropout: float, **kwargs: Any) -> None:
        super().__init__(charset_train, charset_test,
                         batch_size, lr, warmup_pct, weight_decay)
        self.save_hyperparameters()

        self.max_label_length = max_label_length
        self.decode_ar = decode_ar
        self.refine_iters = refine_iters

        self.encoder = Encoder(img_size, patch_size, embed_dim=embed_dim, depth=enc_depth, num_heads=enc_num_heads,
                               mlp_ratio=enc_mlp_ratio)
        decoder_layer = DecoderLayer(
            embed_dim, dec_num_heads, embed_dim * dec_mlp_ratio, dropout)
        self.decoder = Decoder(
            decoder_layer, num_layers=dec_depth, norm=nn.LayerNorm(embed_dim))

        # Perm/attn mask stuff
        self.rng = np.random.default_rng()
        self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num
        self.perm_forward = perm_forward
        self.perm_mirrored = perm_mirrored

        # We don't predict <bos> nor <pad>
        self.head = nn.Linear(embed_dim, len(self.tokenizer) - 2)
        self.text_embed = TokenEmbedding(len(self.tokenizer), embed_dim)

        # Leehakho
        # print(embed_dim)
        # self.lag_queries = nn.Parameter(torch.Tensor(1, max_label_length + 1, embed_dim))

        # +1 for <eos>
        self.pos_queries = nn.Parameter(
            torch.Tensor(1, max_label_length + 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)
        # Encoder has its own init.
        named_apply(partial(init_weights_, exclude=['encoder']), self)
        nn.init.trunc_normal_(self.pos_queries, std=.02)

        # Leehakho
        # self.lan_embedding = language_embedding

    @torch.jit.ignore
    def no_weight_decay(self):
        param_names = {'text_embed.embedding.weight', 'pos_queries'}
        enc_param_names = {'encoder.' +
                           n for n in self.encoder.no_weight_decay()}
        return param_names.union(enc_param_names)

    def encode(self, img: torch.Tensor):
        return self.encoder(img)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[Tensor] = None,
               tgt_padding_mask: Optional[Tensor] = None, tgt_query: Optional[Tensor] = None,
               tgt_query_mask: Optional[Tensor] = None):
        N, L = tgt.shape
        # <bos> stands for the null context. We only supply position information for characters after <bos>.
        null_ctx = self.text_embed(tgt[:, :1])

        tgt_emb = self.pos_queries[:, :L - 1] + self.text_embed(tgt[:, 1:])
        # Leehakho
        # if self.lan_embedding:
        #     tgt_emb = self.pos_queries[:, :L - 1] + self.text_embed(tgt[:, 1:]) + self.lang_embed[:, :L - 1]
        # else:
        #     tgt_emb = self.pos_queries[:, :L - 1] + self.text_embed(tgt[:, 1:])

        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.dropout(tgt_query)
        return self.decoder(tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, tgt_padding_mask)

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        testing = max_length is None
        max_length = self.max_label_length if max_length is None else min(
            max_length, self.max_label_length)
        bs = images.shape[0]
        # +1 for <eos> at end of sequence.
        num_steps = max_length + 1
        memory = self.encode(images)

        # Query positions up to `num_steps`
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)

        # Special case for the forward permutation. Faster than using `generate_attn_masks()`
        tgt_mask = query_mask = torch.triu(torch.full(
            (num_steps, num_steps), float('-inf'), device=self._device), 1)

        if self.decode_ar:
            tgt_in = torch.full((bs, num_steps), self.pad_id,
                                dtype=torch.long, device=self._device)
            tgt_in[:, 0] = self.bos_id

            logits = []
            for i in range(num_steps):
                j = i + 1  # next token index
                # Efficient decoding:
                # Input the context up to the ith token. We use only one query (at position = i) at a time.
                # This works because of the lookahead masking effect of the canonical (forward) AR context.
                # Past tokens have no access to future tokens, hence are fixed once computed.
                tgt_out = self.decode(tgt_in[:, :j], memory, tgt_mask[:j, :j], tgt_query=pos_queries[:, i:j],
                                      tgt_query_mask=query_mask[i:j, :j])
                # the next token probability is in the output's ith token position
                p_i = self.head(tgt_out)
                logits.append(p_i)
                if j < num_steps:
                    # greedy decode. add the next token index to the target input
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)
                    # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                    if testing and (tgt_in == self.eos_id).any(dim=-1).all():
                        break

            logits = torch.cat(logits, dim=1)
        else:
            # No prior context, so input is just <bos>. We query all positions.
            tgt_in = torch.full((bs, 1), self.bos_id,
                                dtype=torch.long, device=self._device)
            tgt_out = self.decode(tgt_in, memory, tgt_query=pos_queries)
            logits = self.head(tgt_out)

        if self.refine_iters:
            # For iterative refinement, we always use a 'cloze' mask.
            # We can derive it from the AR forward mask by unmasking the token context to the right.
            query_mask[torch.triu(torch.ones(
                num_steps, num_steps, dtype=torch.bool, device=self._device), 2)] = 0
            bos = torch.full((bs, 1), self.bos_id,
                             dtype=torch.long, device=self._device)
            for i in range(self.refine_iters):
                # Prior context is the previous output.
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                # mask tokens beyond the first EOS token.
                tgt_padding_mask = (
                    (tgt_in == self.eos_id).int().cumsum(-1) > 0)
                tgt_out = self.decode(tgt_in, memory, tgt_mask, tgt_padding_mask,
                                      tgt_query=pos_queries, tgt_query_mask=query_mask[:, :tgt_in.shape[1]])
                logits = self.head(tgt_out)

        return logits

    def gen_tgt_perms(self, tgt):
        """Generate shared permutations for the whole batch.
           This works because the same attention mask can be used for the shorter sequences
           because of the padding mask.
        """
        # We don't permute the position of BOS, we permute EOS separately
        max_num_chars = tgt.shape[1] - 2
        # Special handling for 1-character sequences
        if max_num_chars == 1:
            return torch.arange(3, device=self._device).unsqueeze(0)
        perms = [torch.arange(max_num_chars, device=self._device)
                 ] if self.perm_forward else []
        # Additional permutations if needed
        max_perms = math.factorial(max_num_chars)
        if self.perm_mirrored:
            max_perms //= 2
        num_gen_perms = min(self.max_gen_perms, max_perms)
        # For 4-char sequences and shorter, we generate all permutations and sample from the pool to avoid collisions
        # Note that this code path might NEVER get executed since the labels in a mini-batch typically exceed 4 chars.
        if max_num_chars < 5:
            # Pool of permutations to sample from. We only need the first half (if complementary option is selected)
            # Special handling for max_num_chars == 4 which correctly divides the pool into the flipped halves
            if max_num_chars == 4 and self.perm_mirrored:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(max_perms))
            perm_pool = torch.as_tensor(list(permutations(
                range(max_num_chars), max_num_chars)), device=self._device)[selector]
            # If the forward permutation is always selected, no need to add it to the pool for sampling
            if self.perm_forward:
                perm_pool = perm_pool[1:]
            perms = torch.stack(perms)
            if len(perm_pool):
                i = self.rng.choice(
                    len(perm_pool), size=num_gen_perms - len(perms), replace=False)
                perms = torch.cat([perms, perm_pool[i]])
        else:
            perms.extend([torch.randperm(max_num_chars, device=self._device)
                         for _ in range(num_gen_perms - len(perms))])
            perms = torch.stack(perms)
        if self.perm_mirrored:
            # Add complementary pairs
            comp = perms.flip(-1)
            # Stack in such a way that the pairs are next to each other.
            perms = torch.stack([perms, comp]).transpose(
                0, 1).reshape(-1, max_num_chars)
        # NOTE:
        # The only meaningful way of permuting the EOS position is by moving it one character position at a time.
        # However, since the number of permutations = T! and number of EOS positions = T + 1, the number of possible EOS
        # positions will always be much less than the number of permutations (unless a low perm_num is set).
        # Thus, it would be simpler to just train EOS using the full and null contexts rather than trying to evenly
        # distribute it across the chosen number of permutations.
        # Add position indices of BOS and EOS
        bos_idx = perms.new_zeros((len(perms), 1))
        eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)
        # Special handling for the reverse direction. This does two things:
        # 1. Reverse context for the characters
        # 2. Null context for [EOS] (required for learning to predict [EOS] in NAR mode)
        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - \
                torch.arange(max_num_chars + 1, device=self._device)
        return perms

    def generate_attn_masks(self, perm):
        """Generate attention masks given a sequence permutation (includes pos. for bos and eos tokens)
        :param perm: the permutation sequence. i = 0 is always the BOS
        :return: lookahead attention masks
        """
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), device=self._device)
        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1:]
            mask[query_idx, masked_keys] = float('-inf')
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=self._device)
             ] = float('-inf')  # mask "self"
        query_mask = mask[1:, :-1]
        return content_mask, query_mask

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        tgt = self.tokenizer.encode(labels, self._device)

        # Encode the source sequence (i.e. the image codes)
        memory = self.encode(images)

        # Prepare the target sequences (input and output)
        tgt_perms = self.gen_tgt_perms(tgt)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        # The [EOS] token is not depended upon by any other token in any permutation ordering
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)

        loss = 0
        loss_numel = 0
        n = (tgt_out != self.pad_id).sum().item()
        for i, perm in enumerate(tgt_perms):
            tgt_mask, query_mask = self.generate_attn_masks(perm)
            out = self.decode(tgt_in, memory, tgt_mask,
                              tgt_padding_mask, tgt_query_mask=query_mask)
            logits = self.head(out).flatten(end_dim=1)
            loss += n * \
                F.cross_entropy(logits, tgt_out.flatten(),
                                ignore_index=self.pad_id)
            loss_numel += n
            # After the second iteration (i.e. done with canonical and reverse orderings),
            # remove the [EOS] tokens for the succeeding perms
            if i == 1:
                tgt_out = torch.where(
                    tgt_out == self.eos_id, self.pad_id, tgt_out)
                n = (tgt_out != self.pad_id).sum().item()
        loss /= loss_numel

        self.log('loss', loss)
        return loss

def load_from_checkpoint(checkpoint_path: str, device):
    model = PARSeq.load_from_checkpoint(checkpoint_path, map_location=device)
    return model

def show_mask(mask, ax, GT=False, random_color=False):
    if type(mask) is torch.Tensor:
        mask = mask.detach().cpu().numpy()
        
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.9])], axis=0)
    elif GT:
        color = np.array([200/255, 200/255, 255/255, 0.6])
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
    return ax

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)

def DeNormalize(image):
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    
    image *= std
    image += mean
    image *= 255.0
    return image

def find_four_corners(binary_mask):
    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the contour with the maximum area (assuming it's the main object)
    try:
        max_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(max_contour, True)
        approx_polygon = cv2.approxPolyDP(max_contour, epsilon, True)
    except:
        return None
    # Ensure that the polygon has four corners
    if len(approx_polygon) == 4:
        return approx_polygon.reshape(-1, 2)
    else:
        return None

def order_points(pts):
    # 네 점의 평균을 계산하여 중심점 찾기
    center = np.mean(pts, axis=0)

    left_pts = pts[np.where(pts[:, 0] < center[0])]
    right_pts = pts[np.where(pts[:, 0] >= center[0])]

    top_left = left_pts[np.argmin(left_pts[:, 1])]
    bottom_left = left_pts[np.argmax(left_pts[:, 1])]

    top_right = right_pts[np.argmin(right_pts[:, 1])]
    bottom_right = right_pts[np.argmax(right_pts[:, 1])]

    ordered_pts = np.array([top_left, top_right, bottom_right, bottom_left])

    return ordered_pts

def extract_center(output_merged):
    pred_points = []
    for i, output_merge in enumerate(output_merged):
        if output_merge is not None:
            for *xyxy, conf, cls in output_merge.tolist():
                x1, y1, x2, y2 = xyxy
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
                pred_points.append([i, conf, x, y])

    return pred_points

def is_long_banner_horizontal(corners):
    min_values = np.min(corners, axis=0)
    max_values = np.max(corners, axis=0)
    
    horizontal_distance = max_values[0] - min_values[0]
    vertical_distance = max_values[1] - min_values[1]
    
    return horizontal_distance > vertical_distance

def banWarp(seg_out, imgs, idx, args):
    warp_out = []
    inverse_transform_matrix = []
    corner_out = []
    warped_idx = []
    for k, pred_mask in enumerate(seg_out):
        img = imgs[k].permute(1, 2, 0).detach().cpu().numpy()
        img = DeNormalize(img).astype(int)

        four_corners = find_four_corners(pred_mask)

        if four_corners is None:
            continue

        four_corners = np.array(four_corners, dtype=np.int32)
        four_corners = order_points(four_corners)
 
        if is_long_banner_horizontal(four_corners):
            output_size = (1536, 512)

            output_image = np.zeros((1536, 512, 3), dtype=np.uint8)

            transform_matrix = cv2.getPerspectiveTransform(
                four_corners.astype(np.float32), np.float32([[0, 0], [1536, 0], [1536, 512], [0, 512]]))
        else:    
            output_size = (512, 640)

            output_image = np.zeros((512, 640, 3), dtype=np.uint8)

            transform_matrix = cv2.getPerspectiveTransform(
                four_corners.astype(np.float32), np.float32([[0, 0], [512, 0], [512, 640], [0, 640]]))

        try:
            inverse_transform_mat = np.linalg.inv(transform_matrix)
        except:
            # transform_matrix is singular
            inverse_transform_mat = np.linalg.pinv(transform_matrix)

        inverse_transform_matrix.append(inverse_transform_mat)

        cv2_img = cv2.cvtColor(
            img.astype(np.uint8), cv2.COLOR_BGR2RGB)

        output_image = cv2.warpPerspective(
            cv2_img, transform_matrix, output_size)

        output_image = cv2.cvtColor(
            output_image.astype(np.uint8), cv2.COLOR_BGR2RGB)

        warp_out.append(output_image)

        corner_out.append(four_corners)

        warped_idx.append(idx[k][0])

        if args.warping.save_results:
            os.makedirs(os.path.join(
                args.save_root_path), exist_ok=True)
            idx_list = [i for i, x in enumerate(warped_idx) if x == idx[k][0]]
            filename = args.out_img_filename.split(".")
            filename[0] += f"_warp_{idx[k][0]}_{len(idx_list)-1}"
            filename = ".".join(filename)
            cv2.imwrite(os.path.join(args.save_root_path, filename),
                        cv2.cvtColor(output_image.astype(np.uint8), cv2.COLOR_RGB2BGR))  
    
    return warp_out, inverse_transform_matrix, corner_out, warped_idx

def preprocess_image(path, params, device="cuda"):
    target_size = (1024, 1024)
    imgs = []
    imgs_array = []
    
    if params.preprocess.save_results:
        if not os.path.exists(os.path.join(params.save_root_path)):
            os.makedirs(os.path.join(params.save_root_path))
    if path.lower().endswith('.mp4'):
        sample_rate = params.sample_rate

        video = cv2.VideoCapture(path)

        fps = video.get(cv2.CAP_PROP_FPS)

        # sample_rate means how many frames to capture in a second.
        frame_interval = int(fps / sample_rate)

        current_frame = 0
        while True:
            # Read next frame
            ret, frame = video.read()

            # No more frames
            if not ret:
                break

            # Select frames to capture
            if current_frame % frame_interval == 0:
                if params.preprocess.save_results:
                    filename = params.out_img_filename.split(".")
                    filename[0] += f"_det{current_frame / frame_interval}"
                    filename = ".".join(filename)
                    cv2.imwrite(os.path.join(params.save_root_path, filename), frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                original_size = frame.shape[:2]
                frame_resized = cv2.resize(frame, target_size)
                
                imgs_array.append(frame)

                frame_resized = frame_resized.astype(np.float32)
                frame_resized = torch.Tensor(
                    frame_resized).permute(2, 0, 1) / 255.

                imgs.append(frame_resized)

            current_frame += 1

        # Release video object
        video.release()
        
    elif path.lower().endswith(('.jpg', '.jpeg', '.png')):
        img = Image.open(path)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_size = img.shape[:2]
        img = cv2.resize(img, target_size)
        imgs_array.append(img)
        img = img.astype(np.float32)
        img = torch.Tensor(img).permute(2, 0, 1) / 255.

        imgs.append(img)

    elif path.lower().endswith('.txt'):
        pass

    else:
        Exception("Invalid data type...")

    imgs = torch.stack(imgs).to(device)

    ratio = (4, 4)
    # ratio = (1, 1)

    return imgs, torch.Tensor(ratio).to(device), original_size, imgs_array

def select_images_based_on_pred_points(imgs, pred_points):
    # 이미지 선택을 위한 인덱스 리스트 생성
    selected_indices = [point[0] for point in pred_points]

    selected_points = [point[2:] for point in pred_points]

    # 선택된 이미지를 저장할 리스트
    selected_images = []

    for index in selected_indices:
        # imgs 텐서에서 해당 인덱스의 이미지를 선택
        selected_image = imgs[index]
        # 선택된 이미지를 리스트에 추가
        selected_images.append(selected_image)

    # 선택된 이미지 리스트를 텐서로 변환
    selected_images_tensor = torch.stack(selected_images)

    selected_points = torch.Tensor(selected_points).unsqueeze(1)

    return selected_images_tensor, selected_points

class LoadImagesOnly(Dataset):
    def __init__(self, path):

        # 이미지 파일 로드
        f = []  # 이미지 파일 리스트
        for p in path if isinstance(path, list) else [path]:
            p = Path(p)  # OS-agnostic 경로 변환
            if p.is_dir():  # 디렉토리인 경우
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)
            elif p.is_file():  # 파일인 경우
                with open(p, 'r', encoding='utf-8') as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    # 로컬 경로를 글로벌 경로로 변환
                    f += [x.replace('./', parent)
                          if x.startswith('./') else x for x in t]
            else:
                raise Exception('%s does not exist' % p)
        self.img_files = sorted(
            [x.replace('/', os.sep) for x in f])
        assert self.img_files, 'No images found'

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # load image
        img_path = self.img_files[index]

        img = Image.open(img_path)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        width, height = img.shape[1], img.shape[0]
        img = cv2.resize(img, (1024, 1024))
        img = img.astype(np.float32)

        w, h = 1024, 1024

        half_size = (w // 4, h // 4)
        img_half = cv2.resize(img, half_size)

        return img, img_half, img_path, torch.Tensor([w / half_size[0], h / half_size[1]]), torch.Tensor([width, height])

def data_loader(args):
    dataset = LoadImagesOnly(args.data)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=None,
        num_workers=8,
        pin_memory=True,
    )

    return dataloader

class SAMFinetuner(pl.LightningModule):

    def __init__(
        self,
        model_type,
        checkpoint_path,
        freeze_image_encoder=False,
        freeze_prompt_encoder=False,
        freeze_mask_decoder=False,
        batch_size=1,
        learning_rate=1e-4,
        weight_decay=1e-4,
        train_dataset=None,
        val_dataset=None,
        metrics_interval=10,
        image_size=1024,
        device="cuda",
    ):
        super(SAMFinetuner, self).__init__()

        self.model_type = model_type
        self.model = sam_model_registry[self.model_type](
            checkpoint=checkpoint_path, device=device)
        self.model.to(device=device)
        self.model.eval()
        self.freeze_image_encoder = freeze_image_encoder
        if freeze_image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if freeze_prompt_encoder:
            for k, param in self.model.prompt_encoder.named_parameters():
                if 'modulate_prompt' in k:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        if freeze_mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_metric = defaultdict(lambda: deque(maxlen=metrics_interval))
        self.val_metric = defaultdict(lambda: deque(maxlen=metrics_interval))

        self.metrics_interval = metrics_interval

        self.transform = ResizeLongestSide(image_size)
        self.image_size = image_size
        self.mask_threshold: float = 3.0
        self.validation_step_outputs = []

    def forward(self, imgs, gt_masks, points, point_labels, gt_polygons, jitter_polygons):
        _, _, H, W = imgs.shape

        features = self.model.image_encoder(imgs)

        loss_focal = loss_dice = loss_iou = 0.
        predictions = []
        confidence_predictions = []
        for feature, point, point_label, gt_mask, gt_polygon, jitter_polygon in zip(features, points, point_labels, gt_masks, gt_polygons, jitter_polygons):

            point_coords = self.transform.apply_coords_torch(
                point, (self.image_size, self.image_size))
            coords_torch = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(
                point_label, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None,
                                                      :, :], labels_torch[None, :]

            point = (coords_torch, labels_torch)

            gt_polygon_torch = torch.as_tensor(
                gt_polygon, dtype=torch.float, device=self.device)
            gt_polygon_torch = gt_polygon_torch[None, :, :]

            jitter_polygon_torch = torch.as_tensor(
                jitter_polygon, dtype=torch.float, device=self.device)
            jitter_polygon_torch = jitter_polygon_torch[None, :, :]

            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                feature=feature,
                points=point,
                boxes=None,
                masks=None,
            )

            # Predict masks
            low_res_masks, iou_predictions, pred_poly = self.model.mask_decoder(
                image_embeddings=feature.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                jitter_poly_torch=jitter_polygon_torch,
            )

            # Upscale the masks to the original image resolution
            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )

            predictions.append(masks)
            confidence_predictions.append(iou_predictions)
        return {
            'loss': 20. * loss_focal + loss_dice + loss_iou,  # SAM default loss
            'loss_focal': loss_focal,
            'loss_dice': loss_dice,
            'loss_iou': loss_iou,
            'predictions': predictions,
            'confidence_predictions': confidence_predictions,
        }

    def validation_step(self, batch, batch_nb):
        outputs = self(batch)

        outputs.pop("confidence_predictions")
        outputs.pop("predictions")

        # validation log
        for metric in ['tp', 'fp', 'fn', 'tn']:
            self.val_metric[metric].append(outputs[metric])
        # aggregate step metics
        step_metrics = [torch.cat(list(self.val_metric[metric]))
                        for metric in ['tp', 'fp', 'fn', 'tn']]
        per_mask_iou = smp.metrics.iou_score(
            *step_metrics, reduction="micro-imagewise")
        self.validation_step_outputs.append(per_mask_iou)
        metrics = {"val_per_mask_iou": per_mask_iou}
        self.log("val_per_mask_iou", per_mask_iou, sync_dist=True)

        return metrics

    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.log("validation_epoch_average iou", epoch_average, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        def warmup_step_lr_builder(warmup_steps, milestones, gamma):
            def warmup_step_lr(steps):
                if steps < warmup_steps:
                    lr_scale = (steps + 1.) / float(warmup_steps)
                else:
                    lr_scale = 1.
                    for milestone in sorted(milestones):
                        if steps >= milestone * self.trainer.estimated_stepping_batches:
                            lr_scale *= gamma
                return lr_scale
            return warmup_step_lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt,
            # warmup change
            warmup_step_lr_builder(250, [0.66667, 0.86666], 0.1)
        )
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': "step",
                'frequency': 1,
            }
        }

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            # collate_fn=self.train_dataset.collate_fn,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            # collate_fn=self.val_dataset.collate_fn,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=False)
        return val_loader

    def test_step(self, batch, batch_nb):
        outputs = self(batch)

        print("TEST STEP PREOCESSED")

        imgs = batch[0]
        gt_masks = batch[1]
        points = batch[2]
        point_labels = batch[3]
        gt_polygons = batch[4]
        jitter_polygons = batch[5]
        image_ids = batch[6]
        pred_masks = outputs['predictions']
        confidence_predictions = outputs['confidence_predictions']
        outputs.pop("predictions")
        outputs.pop("confidence_predictions")

        # # 폴더가 존재하지 않으면 폴더를 생성합니다.
        # if not os.path.exists(self.save_base):
        #     os.makedirs(self.save_base)

        # # TEST image save process
        # for img, gt_mask, pred_mask, point, point_label, image_id, score \
        #         in zip(imgs, gt_masks, pred_masks, points, point_labels, image_ids, confidence_predictions):
        #     img = img.permute(1, 2, 0).detach().cpu().numpy()
        #     img = DeNormalize(img).astype(int)
        #     score = score.detach().cpu().numpy()[0][0]
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(img)
        #     pred_mask = pred_mask > self.mask_threshold
        #     id = image_id.split('.')[0]

        #     show_mask(pred_mask, plt.gca())
        #     point = point.detach().cpu().numpy()
        #     show_points(point, point_label.detach().cpu().numpy(), plt.gca())
        #     plt.axis('off')
        #     filename = f"{id}_pred_{self.mask_threshold:.1f}.png"
        #     plt.savefig(os.path.join(self.save_base, filename),
        #                 bbox_inches='tight', pad_inches=0)
        #     plt.close()

        # validation log
        for metric in ['tp', 'fp', 'fn', 'tn']:
            self.val_metric[metric].append(outputs[metric])
        # aggregate step metics
        step_metrics = [torch.cat(list(self.val_metric[metric]))
                        for metric in ['tp', 'fp', 'fn', 'tn']]
        per_mask_iou = smp.metrics.iou_score(
            *step_metrics, reduction="micro-imagewise")
        self.validation_step_outputs.append(per_mask_iou)
        metrics = {"val_per_mask_iou": per_mask_iou}
        self.log("val_per_mask_iou", per_mask_iou, sync_dist=True)

        return metrics

    def on_test_epoch_end(self):
        print('test_end')
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.log("validation_epoch_average iou", epoch_average, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory

    def inference_step(self, img, points):
        # img should be 1 x 3 x H x W
        _, _, H, W = img.shape

        # points should be 1 x N x 2
        num_points = points.shape[1]

        point_label = torch.ones(
            (1, num_points,), dtype=torch.int, device=self.device)

        random_tmp = np.stack((np.random.randint(0, 64, num_points),
                               np.random.randint(0, 64, num_points)), axis=-1)
        random_tmp = torch.from_numpy(random_tmp).to(self.device)

        random_tmp = random_tmp.unsqueeze(0)

        gt_mask_tmp = torch.zeros((1, 1, H, W), dtype=torch.float32)

        output = self(img, gt_mask_tmp, points,
                      point_label, random_tmp, random_tmp)

        return output

def copy_state_dict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    height, width, channel = img.shape

    # magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > square_size:
        target_size = square_size
    
    ratio = target_size / max(height, width)    

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation = interpolation)


    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w/2), int(target_h/2))

    return resized, ratio, size_heatmap

def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img

def warpCoord(Minv, pt):
    out = np.matmul(Minv, (pt[0], pt[1], 1))
    return np.array([out[0]/out[2], out[1]/out[2]])

def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text):
    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)

    det = []
    mapper = []
    for k in range(1,nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        # thresholding
        if np.max(textmap[labels==k]) < text_threshold: continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        segmap[np.logical_and(link_score==1, text_score==0)] = 0   # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0 : sx = 0
        if sy < 0 : sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:,0]), max(np_contours[:,0])
            t, b = min(np_contours[:,1]), max(np_contours[:,1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4-startidx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)

    return det, labels, mapper

def getPoly_core(boxes, labels, mapper, linkmap):
    # configs
    num_cp = 5
    max_len_ratio = 0.7
    expand_ratio = 1.45
    max_r = 2.0
    step_r = 0.2

    polys = []  
    for k, box in enumerate(boxes):
        # size filter for small instance
        w, h = int(np.linalg.norm(box[0] - box[1]) + 1), int(np.linalg.norm(box[1] - box[2]) + 1)
        if w < 10 or h < 10:
            polys.append(None); continue

        # warp image
        tar = np.float32([[0,0],[w,0],[w,h],[0,h]])
        M = cv2.getPerspectiveTransform(box, tar)
        word_label = cv2.warpPerspective(labels, M, (w, h), flags=cv2.INTER_NEAREST)
        try:
            Minv = np.linalg.inv(M)
        except:
            polys.append(None); continue

        # binarization for selected label
        cur_label = mapper[k]
        word_label[word_label != cur_label] = 0
        word_label[word_label > 0] = 1

        """ Polygon generation """
        # find top/bottom contours
        cp = []
        max_len = -1
        for i in range(w):
            region = np.where(word_label[:,i] != 0)[0]
            if len(region) < 2 : continue
            cp.append((i, region[0], region[-1]))
            length = region[-1] - region[0] + 1
            if length > max_len: max_len = length

        # pass if max_len is similar to h
        if h * max_len_ratio < max_len:
            polys.append(None); continue

        # get pivot points with fixed length
        tot_seg = num_cp * 2 + 1
        seg_w = w / tot_seg     # segment width
        pp = [None] * num_cp    # init pivot points
        cp_section = [[0, 0]] * tot_seg
        seg_height = [0] * num_cp
        seg_num = 0
        num_sec = 0
        prev_h = -1
        for i in range(0,len(cp)):
            (x, sy, ey) = cp[i]
            if (seg_num + 1) * seg_w <= x and seg_num <= tot_seg:
                # average previous segment
                if num_sec == 0: break
                cp_section[seg_num] = [cp_section[seg_num][0] / num_sec, cp_section[seg_num][1] / num_sec]
                num_sec = 0

                # reset variables
                seg_num += 1
                prev_h = -1

            # accumulate center points
            cy = (sy + ey) * 0.5
            cur_h = ey - sy + 1
            cp_section[seg_num] = [cp_section[seg_num][0] + x, cp_section[seg_num][1] + cy]
            num_sec += 1

            if seg_num % 2 == 0: continue # No polygon area

            if prev_h < cur_h:
                pp[int((seg_num - 1)/2)] = (x, cy)
                seg_height[int((seg_num - 1)/2)] = cur_h
                prev_h = cur_h

        # processing last segment
        if num_sec != 0:
            cp_section[-1] = [cp_section[-1][0] / num_sec, cp_section[-1][1] / num_sec]

        # pass if num of pivots is not sufficient or segment widh is smaller than character height 
        if None in pp or seg_w < np.max(seg_height) * 0.25:
            polys.append(None); continue

        # calc median maximum of pivot points
        half_char_h = np.median(seg_height) * expand_ratio / 2

        # calc gradiant and apply to make horizontal pivots
        new_pp = []
        for i, (x, cy) in enumerate(pp):
            dx = cp_section[i * 2 + 2][0] - cp_section[i * 2][0]
            dy = cp_section[i * 2 + 2][1] - cp_section[i * 2][1]
            if dx == 0:     # gradient if zero
                new_pp.append([x, cy - half_char_h, x, cy + half_char_h])
                continue
            rad = - math.atan2(dy, dx)
            c, s = half_char_h * math.cos(rad), half_char_h * math.sin(rad)
            new_pp.append([x - s, cy - c, x + s, cy + c])

        # get edge points to cover character heatmaps
        isSppFound, isEppFound = False, False
        grad_s = (pp[1][1] - pp[0][1]) / (pp[1][0] - pp[0][0]) + (pp[2][1] - pp[1][1]) / (pp[2][0] - pp[1][0])
        grad_e = (pp[-2][1] - pp[-1][1]) / (pp[-2][0] - pp[-1][0]) + (pp[-3][1] - pp[-2][1]) / (pp[-3][0] - pp[-2][0])
        for r in np.arange(0.5, max_r, step_r):
            dx = 2 * half_char_h * r
            if not isSppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_s * dx
                p = np.array(new_pp[0]) - np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    spp = p
                    isSppFound = True
            if not isEppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_e * dx
                p = np.array(new_pp[-1]) + np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    epp = p
                    isEppFound = True
            if isSppFound and isEppFound:
                break

        # pass if boundary of polygon is not found
        if not (isSppFound and isEppFound):
            polys.append(None); continue

        # make final polygon
        poly = []
        poly.append(warpCoord(Minv, (spp[0], spp[1])))
        for p in new_pp:
            poly.append(warpCoord(Minv, (p[0], p[1])))
        poly.append(warpCoord(Minv, (epp[0], epp[1])))
        poly.append(warpCoord(Minv, (epp[2], epp[3])))
        for p in reversed(new_pp):
            poly.append(warpCoord(Minv, (p[2], p[3])))
        poly.append(warpCoord(Minv, (spp[2], spp[3])))

        # add to final result
        polys.append(np.array(poly))

    return polys

def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly=False):
    boxes, labels, mapper = getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text)

    if poly:
        polys = getPoly_core(boxes, labels, mapper, linkmap)
    else:
        polys = [None] * len(boxes)

    return boxes, polys

def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net = 2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys

def test_net(net, image, text_threshold, link_threshold, low_text, poly, args, gpu_id, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
        image, args.scene_text_detection.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.scene_text_detection.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if gpu_id is not None:
        x = x.cuda(gpu_id)

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text

def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

class vgg16_bn(torch.nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super(vgg16_bn, self).__init__()
        # model_urls['vgg16_bn'] = model_urls['vgg16_bn'].replace(
        #     'https://', 'http://')
        vgg_pretrained_features = models.vgg16_bn(
            pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(12):         # conv2_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):         # conv3_3
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 29):         # conv4_3
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(29, 39):         # conv5_3
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        # fc6, fc7 without atrous conv
        self.slice5 = torch.nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1)
        )

        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())

        # no pretrained model for fc6 and fc7
        init_weights(self.slice5.modules())

        if freeze:
            for param in self.slice1.parameters():      # only first conv
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out

class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class CRAFT(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()

        """ Base network """
        self.basenet = vgg16_bn(pretrained, freeze)

        """ U network """
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        """ Base network """
        sources = self.basenet(x)

        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[
                          2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[
                          2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[
                          2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0, 2, 3, 1), feature

def print_gpu_usage(device1, device2):
    print(f"Device1 Memory Allocated: {torch.cuda.memory_allocated(device1) / 1024 ** 2:.2f} MB")
    print(f"Device1 Memory Reserved: {torch.cuda.memory_reserved(device1) / 1024 ** 2:.2f} MB")
    print(f"Device2 Memory Allocated: {torch.cuda.memory_allocated(device2) / 1024 ** 2:.2f} MB")
    print(f"Device2 Memory Reserved: {torch.cuda.memory_reserved(device2) / 1024 ** 2:.2f} MB")

def load_det_model(gpu_id, model_path):
    
    config_det = Config(cfg_dict={'auto_scale_lr': {'base_batch_size': 64, 'enable': False}, 'backend_args': None, 'class_name': ('outdoor banner',), 'coco_od_dataset': {'ann_file': 'o365v1_train_odvg.json', 'backend_args': None, 'data_prefix': {'img': 'train/'}, 'data_root': 'data/objects365v1/', 'filter_cfg': {'filter_empty_gt': False}, 'label_map_file': 'o365v1_label_map.json', 'pipeline': [{'backend_args': None, 'type': 'LoadImageFromFile'}, {'type': 'LoadAnnotations', 'with_bbox': True}, {'prob': 0.5, 'type': 'RandomFlip'}, {'transforms': [[{'keep_ratio': True, 'scales': [(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333), (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333)], 'type': 'RandomChoiceResize'}], [{'keep_ratio': True, 'scales': [(400, 4200), (500, 4200), (600, 4200)], 'type': 'RandomChoiceResize'}, {'allow_negative_crop': True, 'crop_size': (384, 600), 'crop_type': 'absolute_range', 'type': 'RandomCrop'}, {'keep_ratio': True, 'scales': [(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333), (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333)], 'type': 'RandomChoiceResize'}]], 'type': 'RandomChoice'}, {'min_gt_bbox_wh': (0.01, 0.01), 'type': 'FilterAnnotations'}, {'max_tokens': 256, 'num_sample_negative': 85, 'tokenizer_name': 'bert-base-uncased', 'type': 'RandomSamplingNegPos'}, {'meta_keys': ('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'text', 'custom_entities', 'tokens_positive', 'dataset_mode'), 'type': 'PackDetInputs'}], 'return_classes': True, 'type': 'ODVGDataset'}, 'data_root': 'data/REAL/', 'dataset_type': 'ODVGDataset', 'default_hooks': {'checkpoint': {'interval': 1, 'max_keep_ckpts': 1, 'save_best': 'auto', 'type': 'CheckpointHook'}, 'logger': {'interval': 5, 'type': 'LoggerHook'}, 'param_scheduler': {'type': 'ParamSchedulerHook'}, 'sampler_seed': {'type': 'DistSamplerSeedHook'}, 'timer': {'type': 'IterTimerHook'}, 'visualization': {'type': 'GroundingVisualizationHook'}}, 'default_scope': 'mmdet', 'env_cfg': {'cudnn_benchmark': False, 'dist_cfg': {'backend': 'nccl'}, 'mp_cfg': {'mp_start_method': 'fork', 'opencv_num_threads': 0}}, 'lang_model_name': 'bert-base-uncased', 'launcher': 'pytorch', 'load_from': '/home/intern/fine_tune_GroundingDINO/mmdetection/auto_label_v1/best_coco_bbox_mAP_epoch_6.pth', 'log_level': 'INFO', 'log_processor': {'by_epoch': True, 'type': 'LogProcessor', 'window_size': 50}, 'max_epoch': 20, 'max_epochs': 30, 'metainfo': {'classes': ('outdoor banner',), 'palette': [(220, 20, 60)]}, 'model': {'as_two_stage': True, 'backbone': {'attn_drop_rate': 0.0, 'convert_weights': True, 'depths': [2, 2, 6, 2], 'drop_path_rate': 0.2, 'drop_rate': 0.0, 'embed_dims': 96, 'frozen_stages': -1, 'init_cfg': {'checkpoint': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth', 'type': 'Pretrained'}, 'mlp_ratio': 4, 'num_heads': [3, 6, 12, 24], 'out_indices': (1, 2, 3), 'patch_norm': True, 'qk_scale': None, 'qkv_bias': True, 'type': 'SwinTransformer', 'window_size': 7, 'with_cp': True}, 'bbox_head': {'contrastive_cfg': {'bias': True, 'log_scale': 'auto', 'max_text_len': 256}, 'loss_bbox': {'loss_weight': 5.0, 'type': 'L1Loss'}, 'loss_cls': {'alpha': 0.25, 'gamma': 2.0, 'loss_weight': 1.0, 'type': 'FocalLoss', 'use_sigmoid': True}, 'num_classes': 1, 'sync_cls_avg_factor': True, 'type': 'GroundingDINOHead'}, 'data_preprocessor': {'bgr_to_rgb': True, 'mean': [123.675, 116.28, 103.53], 'pad_mask': False, 'std': [58.395, 57.12, 57.375], 'type': 'DetDataPreprocessor'}, 'decoder': {'layer_cfg': {'cross_attn_cfg': {'dropout': 0.0, 'embed_dims': 256, 'num_heads': 8}, 'cross_attn_text_cfg': {'dropout': 0.0, 'embed_dims': 256, 'num_heads': 8}, 'ffn_cfg': {'embed_dims': 256, 'feedforward_channels': 2048, 'ffn_drop': 0.0}, 'self_attn_cfg': {'dropout': 0.0, 'embed_dims': 256, 'num_heads': 8}}, 'num_layers': 6, 'post_norm_cfg': None, 'return_intermediate': True}, 'dn_cfg': {'box_noise_scale': 1.0, 'group_cfg': {'dynamic': True, 'num_dn_queries': 100, 'num_groups': None}, 'label_noise_scale': 0.5}, 'encoder': {'fusion_layer_cfg': {'embed_dim': 1024, 'init_values': 0.0001, 'l_dim': 256, 'num_heads': 4, 'v_dim': 256}, 'layer_cfg': {'ffn_cfg': {'embed_dims': 256, 'feedforward_channels': 2048, 'ffn_drop': 0.0}, 'self_attn_cfg': {'dropout': 0.0, 'embed_dims': 256, 'num_levels': 4}}, 'num_cp': 6, 'num_layers': 6, 'text_layer_cfg': {'ffn_cfg': {'embed_dims': 256, 'feedforward_channels': 1024, 'ffn_drop': 0.0}, 'self_attn_cfg': {'dropout': 0.0, 'embed_dims': 256, 'num_heads': 4}}}, 'language_model': {'add_pooling_layer': False, 'max_tokens': 256, 'name': 'bert-base-uncased', 'pad_to_max': False, 'special_tokens_list': ['[CLS]', '[SEP]', '.', '?'], 'type': 'BertModel', 'use_sub_sentence_represent': True}, 'neck': {'act_cfg': None, 'bias': True, 'in_channels': [192, 384, 768], 'kernel_size': 1, 'norm_cfg': {'num_groups': 32, 'type': 'GN'}, 'num_outs': 4, 'out_channels': 256, 'type': 'ChannelMapper'}, 'num_queries': 900, 'positional_encoding': {'normalize': True, 'num_feats': 128, 'offset': 0.0, 'temperature': 20}, 'test_cfg': {'max_per_img': 300}, 'train_cfg': {'assigner': {'match_costs': [{'type': 'BinaryFocalLossCost', 'weight': 0.2222222222222222}, {'box_format': 'xywh', 'type': 'BBoxL1Cost', 'weight': 0.5555555555555556}, {'iou_mode': 'giou', 'type': 'IoUCost', 'weight': 0.2222222222222222}], 'type': 'HungarianAssigner'}}, 'type': 'GroundingDINO', 'with_box_refine': True}, 'num_classes': 1, 'optim_wrapper': {'clip_grad': {'max_norm': 0.1, 'norm_type': 2}, 'optimizer': {'lr': 0.0001, 'type': 'AdamW', 'weight_decay': 0.0001}, 'paramwise_cfg': {'custom_keys': {'absolute_pos_embed': {'decay_mult': 0.0}, 'backbone': {'lr_mult': 0.0}, 'language_model': {'lr_mult': 0.0}}}, 'type': 'OptimWrapper'}, 'param_scheduler': [{'begin': 0, 'by_epoch': True, 'end': 20, 'gamma': 0.1, 'milestones': [15], 'type': 'MultiStepLR'}], 'pretrained': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth', 'resume': False, 'test_cfg': {'type': 'TestLoop'}, 'test_dataloader': {'batch_size': 1, 'dataset': {'ann_file': 'annotations/valid.json', 'backend_args': None, 'data_prefix': {'img': 'images/'}, 'data_root': 'data/REAL/', 'metainfo': {'classes': ('outdoor banner',), 'palette': [(220, 20, 60)]}, 'pipeline': [{'backend_args': None, 'imdecode_backend': 'pillow', 'type': 'LoadImageFromFile'}, {'backend': 'pillow', 'keep_ratio': True, 'scale': (800, 1333), 'type': 'FixScaleResize'}, {'type': 'LoadAnnotations', 'with_bbox': True}, {'meta_keys': ('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'text', 'custom_entities', 'tokens_positive'), 'type': 'PackDetInputs'}], 'return_classes': True, 'test_mode': True, 'type': 'CocoDataset'}, 'drop_last': False, 'num_workers': 2, 'persistent_workers': True, 'sampler': {'shuffle': False, 'type': 'DefaultSampler'}}, 'test_evaluator': {'ann_file': 'data/REAL/annotations/valid.json', 'backend_args': None, 'format_only': False, 'metric': 'bbox', 'type': 'CocoMetric'}, 'test_pipeline': [{'backend_args': None, 'imdecode_backend': 'pillow', 'type': 'LoadImageFromFile'}, {'backend': 'pillow', 'keep_ratio': True, 'scale': (800, 1333), 'type': 'FixScaleResize'}, {'type': 'LoadAnnotations', 'with_bbox': True}, {'meta_keys': ('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'text', 'custom_entities', 'tokens_positive'), 'type': 'PackDetInputs'}], 'train_cfg': {'max_epochs': 20, 'type': 'EpochBasedTrainLoop', 'val_interval': 1}, 'train_dataloader': {'batch_sampler': {'type': 'AspectRatioBatchSampler'}, 'batch_size': 4, 'dataset': {'ann_file': 'annotations/train_real_v1.json', 'data_prefix': {'img': 'images/'}, 'data_root': 'data/REAL/', 'filter_cfg': {'filter_empty_gt': False, 'min_size': 32}, 'metainfo': {'classes': ('outdoor banner',), 'palette': [(220, 20, 60)]}, 'pipeline': [{'type': 'LoadImageFromFile'}, {'type': 'LoadAnnotations', 'with_bbox': True}, {'prob': 0.5, 'type': 'RandomFlip'}, {'transforms': [[{'keep_ratio': True, 'scales': [(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333), (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333)], 'type': 'RandomChoiceResize'}], [{'keep_ratio': True, 'scales': [(400, 4200), (500, 4200), (600, 4200)], 'type': 'RandomChoiceResize'}, {'allow_negative_crop': True, 'crop_size': (384, 600), 'crop_type': 'absolute_range', 'type': 'RandomCrop'}, {'keep_ratio': True, 'scales': [(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333), (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333)], 'type': 'RandomChoiceResize'}]], 'type': 'RandomChoice'}, {'meta_keys': ('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'text', 'custom_entities'), 'type': 'PackDetInputs'}], 'return_classes': True, 'type': 'CocoDataset'}, 'num_workers': 4, 'persistent_workers': True, 'sampler': {'shuffle': True, 'type': 'DefaultSampler'}}, 'train_pipeline': [{'type': 'LoadImageFromFile'}, {'type': 'LoadAnnotations', 'with_bbox': True}, {'prob': 0.5, 'type': 'RandomFlip'}, {'transforms': [[{'keep_ratio': True, 'scales': [(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333), (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333)], 'type': 'RandomChoiceResize'}], [{'keep_ratio': True, 'scales': [(400, 4200), (500, 4200), (600, 4200)], 'type': 'RandomChoiceResize'}, {'allow_negative_crop': True, 'crop_size': (384, 600), 'crop_type': 'absolute_range', 'type': 'RandomCrop'}, {'keep_ratio': True, 'scales': [(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333), (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333)], 'type': 'RandomChoiceResize'}]], 'type': 'RandomChoice'}, {'meta_keys': ('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'text', 'custom_entities'), 'type': 'PackDetInputs'}], 'val_cfg': {'type': 'ValLoop'}, 'val_dataloader': {'batch_size': 1, 'dataset': {'ann_file': 'annotations/valid.json', 'backend_args': None, 'data_prefix': {'img': 'images/'}, 'data_root': 'data/REAL/', 'metainfo': {'classes': ('outdoor banner',), 'palette': [(220, 20, 60)]}, 'pipeline': [{'backend_args': None, 'imdecode_backend': 'pillow', 'type': 'LoadImageFromFile'}, {'backend': 'pillow', 'keep_ratio': True, 'scale': (800, 1333), 'type': 'FixScaleResize'}, {'type': 'LoadAnnotations', 'with_bbox': True}, {'meta_keys': ('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'text', 'custom_entities', 'tokens_positive'), 'type': 'PackDetInputs'}], 'return_classes': True, 'test_mode': True, 'type': 'CocoDataset'}, 'drop_last': False, 'num_workers': 2, 'persistent_workers': True, 'sampler': {'shuffle': False, 'type': 'DefaultSampler'}}, 'val_evaluator': {'ann_file': 'data/REAL/annotations/valid.json', 'backend_args': None, 'format_only': False, 'metric': 'bbox', 'type': 'CocoMetric'}, 'vis_backends': [{'type': 'LocalVisBackend'}], 'visualizer': {'name': 'visualizer', 'type': 'DetLocalVisualizer', 'vis_backends': [{'type': 'LocalVisBackend'}]}, 'work_dir': 'auto_label_adaptive_final'})
    
    det_model = init_detector(config_det, model_path, device=gpu_id)
    det_model.eval()
    det_model.cfg.test_dataloader.dataset.pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = compose_cv(det_model.cfg.test_dataloader.dataset.pipeline)
    
    return det_model, test_pipeline
    
    # device = f"cuda:{gpu_id}"
    # det_model = attempt_load(
    #     model_path, map_location=device)
    # det_model.eval()
    # return det_model 

def load_sam_model(gpu_id, model_path):
    device = f"cuda:{gpu_id}"
    sam_model = SAMFinetuner(
        model_type="vit_h",
        checkpoint_path=model_path,
        freeze_image_encoder=True,
        freeze_prompt_encoder=True,
        freeze_mask_decoder=True,
        batch_size=1,
        learning_rate=0.001,
        weight_decay=0.01,
        train_dataset=None,
        val_dataset=None,
        metrics_interval=10,
        image_size=1024,
        device=device
    )
    sam_model.to(device).eval()
    return sam_model

def load_std_model(gpu_id, model_path):
    device = f"cuda:{gpu_id}"
    craft_model = CRAFT()
    craft_model.load_state_dict(
        copy_state_dict(torch.load(model_path, map_location=device)))
    craft_model = craft_model.cuda(gpu_id).eval()
    return craft_model

def load_str_model(gpu_id, model_path):
    device = f"cuda:{gpu_id}"
    charset_test = "0123456789abcdefghijklmnopqrstuvwxyz!$%&()?~가각간갇갈갉갊감갑값갓갔강갖갗같갚갛개객갠갤갬갭갯갰갱갸갹갼걀걋걍걔걘걜거걱건걷걸걺검겁것겄겅겆겉겊겋게겐겔겜겝겟겠겡겨격겪견겯결겸겹겻겼경곁계곈곌곕곗고곡곤곧골곪곬곯곰곱곳공곶과곽관괄괆괌괍괏광괘괜괠괩괬괭괴괵괸괼굄굅굇굉교굔굘굡굣구국군굳굴굵굶굻굼굽굿궁궂궈궉권궐궜궝궤궷귀귁귄귈귐귑귓규균귤그극근귿글긁금급긋긍긔기긱긴긷길긺김깁깃깅깆깊까깍깎깐깔깖깜깝깟깠깡깥깨깩깬깰깸깹깻깼깽꺄꺅꺌꺼꺽꺾껀껄껌껍껏껐껑께껙껜껨껫껭껴껸껼꼇꼈꼍꼐꼬꼭꼰꼲꼴꼼꼽꼿꽁꽂꽃꽈꽉꽐꽜꽝꽤꽥꽹꾀꾄꾈꾐꾑꾕꾜꾸꾹꾼꿀꿇꿈꿉꿋꿍꿎꿔꿜꿨꿩꿰꿱꿴꿸뀀뀁뀄뀌뀐뀔뀜뀝뀨끄끅끈끊끌끎끓끔끕끗끙끝끼끽낀낄낌낍낏낑나낙낚난낟날낡낢남납낫났낭낮낯낱낳내낵낸낼냄냅냇냈냉냐냑냔냘냠냥너넉넋넌널넒넓넘넙넛넜넝넣네넥넨넬넴넵넷넸넹녀녁년녈념녑녔녕녘녜녠노녹논놀놂놈놉놋농높놓놔놘놜놨뇌뇐뇔뇜뇝뇟뇨뇩뇬뇰뇹뇻뇽누눅눈눋눌눔눕눗눙눠눴눼뉘뉜뉠뉨뉩뉴뉵뉼늄늅늉느늑는늘늙늚늠늡늣능늦늪늬늰늴니닉닌닐닒님닙닛닝닢다닥닦단닫달닭닮닯닳담답닷닸당닺닻닿대댁댄댈댐댑댓댔댕댜더덕덖던덛덜덞덟덤덥덧덩덫덮데덱덴델뎀뎁뎃뎄뎅뎌뎐뎔뎠뎡뎨뎬도독돈돋돌돎돐돔돕돗동돛돝돠돤돨돼됐되된될됨됩됫됬됴두둑둔둘둠둡둣둥둬뒀뒈뒝뒤뒨뒬뒵뒷뒹듀듄듈듐듕드득든듣들듦듬듭듯등듸디딕딘딛딜딤딥딧딨딩딪따딱딴딸땀땁땃땄땅땋때땍땐땔땜땝땟땠땡떠떡떤떨떪떫떰떱떳떴떵떻떼떽뗀뗄뗌뗍뗏뗐뗑뗘뗬또똑똔똘똥똬똴뙈뙤뙨뚜뚝뚠뚤뚫뚬뚱뛔뛰뛴뛸뜀뜁뜅뜨뜩뜬뜯뜰뜸뜹뜻띄띈띌띔띕띠띤띨띰띱띳띵라락란랄람랍랏랐랑랒랖랗래랙랜랠램랩랫랬랭랴략랸럇량러럭런럴럼럽럿렀렁렇레렉렌렐렘렙렛렝려력련렬렴렵렷렸령례롄롑롓로록론롤롬롭롯롱롸롼뢍뢨뢰뢴뢸룀룁룃룅료룐룔룝룟룡루룩룬룰룸룹룻룽뤄뤘뤠뤼뤽륀륄륌륏륑류륙륜률륨륩륫륭르륵른를름릅릇릉릊릍릎리릭린릴림립릿링마막만많맏말맑맒맘맙맛망맞맡맣매맥맨맬맴맵맷맸맹맺먀먁먈먕머먹먼멀멂멈멉멋멍멎멓메멕멘멜멤멥멧멨멩며멱면멸몃몄명몇몌모목몫몬몰몲몸몹못몽뫄뫈뫘뫙뫼묀묄묍묏묑묘묜묠묩묫무묵묶문묻물묽묾뭄뭅뭇뭉뭍뭏뭐뭔뭘뭡뭣뭬뮈뮌뮐뮤뮨뮬뮴뮷므믄믈믐믓미믹민믿밀밂밈밉밋밌밍및밑바박밖밗반받발밝밞밟밤밥밧방밭배백밴밸뱀뱁뱃뱄뱅뱉뱌뱍뱐뱝버벅번벋벌벎범법벗벙벚베벡벤벧벨벰벱벳벴벵벼벽변별볍볏볐병볕볘볜보복볶본볼봄봅봇봉봐봔봤봬뵀뵈뵉뵌뵐뵘뵙뵤뵨부북분붇불붉붊붐붑붓붕붙붚붜붤붰붸뷔뷕뷘뷜뷩뷰뷴뷸븀븃븅브븍븐블븜븝븟비빅빈빌빎빔빕빗빙빚빛빠빡빤빨빪빰빱빳빴빵빻빼빽뺀뺄뺌뺍뺏뺐뺑뺘뺙뺨뻐뻑뻔뻗뻘뻠뻣뻤뻥뻬뼁뼈뼉뼘뼙뼛뼜뼝뽀뽁뽄뽈뽐뽑뽕뾔뾰뿅뿌뿍뿐뿔뿜뿟뿡쀼쁑쁘쁜쁠쁨쁩삐삑삔삘삠삡삣삥사삭삯산삳살삵삶삼삽삿샀상샅새색샌샐샘샙샛샜생샤샥샨샬샴샵샷샹섀섄섈섐섕서석섞섟선섣설섦섧섬섭섯섰성섶세섹센셀셈셉셋셌셍셔셕션셜셤셥셧셨셩셰셴셸솅소속솎손솔솖솜솝솟송솥솨솩솬솰솽쇄쇈쇌쇔쇗쇘쇠쇤쇨쇰쇱쇳쇼쇽숀숄숌숍숏숑수숙순숟술숨숩숫숭숯숱숲숴쉈쉐쉑쉔쉘쉠쉥쉬쉭쉰쉴쉼쉽쉿슁슈슉슐슘슛슝스슥슨슬슭슴습슷승시식신싣실싫심십싯싱싶싸싹싻싼쌀쌈쌉쌌쌍쌓쌔쌕쌘쌜쌤쌥쌨쌩썅써썩썬썰썲썸썹썼썽쎄쎈쎌쏀쏘쏙쏜쏟쏠쏢쏨쏩쏭쏴쏵쏸쐈쐐쐤쐬쐰쐴쐼쐽쑈쑤쑥쑨쑬쑴쑵쑹쒀쒔쒜쒸쒼쓩쓰쓱쓴쓸쓺쓿씀씁씌씐씔씜씨씩씬씰씸씹씻씽아악안앉않알앍앎앓암압앗았앙앝앞애액앤앨앰앱앳앴앵야약얀얄얇얌얍얏양얕얗얘얜얠얩어억언얹얻얼얽얾엄업없엇었엉엊엌엎에엑엔엘엠엡엣엥여역엮연열엶엷염엽엾엿였영옅옆옇예옌옐옘옙옛옜오옥온올옭옮옰옳옴옵옷옹옻와왁완왈왐왑왓왔왕왜왝왠왬왯왱외왹왼욀욈욉욋욍요욕욘욜욤욥욧용우욱운울욹욺움웁웃웅워웍원월웜웝웠웡웨웩웬웰웸웹웽위윅윈윌윔윕윗윙유육윤율윰윱윳융윷으윽은을읊음읍읏응읒읓읔읕읖읗의읜읠읨읫이익인일읽읾잃임입잇있잉잊잎자작잔잖잗잘잚잠잡잣잤장잦재잭잰잴잼잽잿쟀쟁쟈쟉쟌쟎쟐쟘쟝쟤쟨쟬저적전절젊점접젓정젖제젝젠젤젬젭젯젱져젼졀졈졉졌졍졔조족존졸졺좀좁좃종좆좇좋좌좍좔좝좟좡좨좼좽죄죈죌죔죕죗죙죠죡죤죵주죽준줄줅줆줌줍줏중줘줬줴쥐쥑쥔쥘쥠쥡쥣쥬쥰쥴쥼즈즉즌즐즘즙즛증지직진짇질짊짐집짓징짖짙짚짜짝짠짢짤짧짬짭짯짰짱째짹짼쨀쨈쨉쨋쨌쨍쨔쨘쨩쩌쩍쩐쩔쩜쩝쩟쩠쩡쩨쩽쪄쪘쪼쪽쫀쫄쫌쫍쫏쫑쫓쫘쫙쫠쫬쫴쬈쬐쬔쬘쬠쬡쭁쭈쭉쭌쭐쭘쭙쭝쭤쭸쭹쮜쮸쯔쯤쯧쯩찌찍찐찔찜찝찡찢찧차착찬찮찰참찹찻찼창찾채책챈챌챔챕챗챘챙챠챤챦챨챰챵처척천철첨첩첫첬청체첵첸첼쳄쳅쳇쳉쳐쳔쳤쳬쳰촁초촉촌촐촘촙촛총촤촨촬촹최쵠쵤쵬쵭쵯쵱쵸춈추축춘출춤춥춧충춰췄췌췐취췬췰췸췹췻췽츄츈츌츔츙츠측츤츨츰츱츳층치칙친칟칠칡침칩칫칭카칵칸칼캄캅캇캉캐캑캔캘캠캡캣캤캥캬캭컁커컥컨컫컬컴컵컷컸컹케켁켄켈켐켑켓켕켜켠켤켬켭켯켰켱켸코콕콘콜콤콥콧콩콰콱콴콸쾀쾅쾌쾡쾨쾰쿄쿠쿡쿤쿨쿰쿱쿳쿵쿼퀀퀄퀑퀘퀭퀴퀵퀸퀼큄큅큇큉큐큔큘큠크큭큰클큼큽킁키킥킨킬킴킵킷킹타탁탄탈탉탐탑탓탔탕태택탠탤탬탭탯탰탱탸턍터턱턴털턺텀텁텃텄텅테텍텐텔템텝텟텡텨텬텼톄톈토톡톤톨톰톱톳통톺톼퇀퇘퇴퇸툇툉툐투툭툰툴툼툽툿퉁퉈퉜퉤튀튁튄튈튐튑튕튜튠튤튬튱트특튼튿틀틂틈틉틋틔틘틜틤틥티틱틴틸팀팁팃팅파팍팎판팔팖팜팝팟팠팡팥패팩팬팰팸팹팻팼팽퍄퍅퍼퍽펀펄펌펍펏펐펑페펙펜펠펨펩펫펭펴편펼폄폅폈평폐폘폡폣포폭폰폴폼폽폿퐁퐈퐝푀푄표푠푤푭푯푸푹푼푿풀풂품풉풋풍풔풩퓌퓐퓔퓜퓟퓨퓬퓰퓸퓻퓽프픈플픔픕픗피픽핀필핌핍핏핑하학한할핥함합핫항핳해핵핸핼햄햅햇했행햐향허헉헌헐헒험헙헛헝헤헥헨헬헴헵헷헹혀혁현혈혐협혓혔형혜혠혤혭호혹혼홀홅홈홉홋홍홑화확환활홧황홰홱홴횃횅회획횐횔횝횟횡효횬횰횹횻후훅훈훌훑훔훗훙훠훤훨훰훵훼훽휀휄휑휘휙휜휠휨휩휫휭휴휵휸휼흄흇흉흐흑흔흖흗흘흙흠흡흣흥흩희흰흴흼흽힁히힉힌힐힘힙힛힝힣"
    charset_test += string.ascii_uppercase
    str_model = load_from_checkpoint(model_path, device).to(device).eval()
    return str_model

def write_to_file(contours, file_path):
    """
    :param contours: [[x1, y1], [x2, y2]... [xn, yn]]
    :param file_path: target file path
    """
    # according to total-text evaluation method, output file shoud be formatted to: y0,x0, ..... yn,xn
    with open(file_path, 'w') as f:
        for cont in contours:
            cont = np.stack([cont[:, 0], cont[:, 1]], 1)
            if cv2.contourArea(cont) <= 0:
                continue
            cont = cont.flatten().astype(str).tolist()
            cont = ','.join(cont)
            f.write(cont + '\n')

def visualize_gt(image, contours, label_tag):

    image_show = image.copy()
    image_show = np.ascontiguousarray(image_show[:, :, ::-1])

    image_show = cv2.polylines(image_show,
                               [contours[i] for i, tag in enumerate(label_tag) if tag >0], True, (0, 0, 255), 3)
    image_show = cv2.polylines(image_show,
                               [contours[i] for i, tag in enumerate(label_tag) if tag <0], True, (0, 255, 0), 3)

    show_gt = cv2.resize(image_show, (320, 320))

    return show_gt

def heatmap(im_gray):
    cmap = plt.get_cmap('jet')
    rgba_img = cmap(255 - im_gray)
    Hmap = np.delete(rgba_img, 3, 2)
    return Hmap

def visualize_detection(image, output_dir, output_dict, meta=None):
    image_show = image.copy()
    image_show = np.ascontiguousarray(image_show[:, :, ::-1])

    cls_preds = F.interpolate(output_dict["fy_preds"], scale_factor=1, mode='bilinear')
    cls_preds = cls_preds[0].data.cpu().numpy()

    py_preds = output_dict["py_preds"][1:]
    shows = []

    im_show0 = image_show.copy()
    for idx, py in enumerate(py_preds):
        im_show = im_show0.copy()
        contours = py.data.cpu().numpy()
        cv2.drawContours(im_show, contours.astype(np.int32), -1, (0, 0, 255), 2)
        for ppts in contours:
            for j, pp in enumerate(ppts):
                if j == 0:
                    cv2.circle(im_show, (int(pp[0]), int(pp[1])), 3, (255, 0, 255), -1)
                elif j == 1:
                    cv2.circle(im_show, (int(pp[0]), int(pp[1])), 3, (0, 255, 255), -1)
                else:
                    cv2.circle(im_show, (int(pp[0]), int(pp[1])), 3, (0, 255, 0), -1)
        path = os.path.join(output_dir, "{}iter.png".format(idx))
        # cv2.imwrite(path, im_show)
        shows.append(im_show)

    show_img = np.concatenate(shows, axis=1)
    show_boundary = cv2.resize(show_img, (320 * len(py_preds), 320))

    cls_pred = heatmap(np.array(cls_preds[0] * 255, dtype=np.uint8))
    dis_pred = heatmap(np.array(cls_preds[1] * 255, dtype=np.uint8))

    heat_map = np.concatenate([cls_pred*255, dis_pred*255], axis=1)
    heat_map = cv2.resize(heat_map, (320 * 2, 320))

    return show_boundary, heat_map

def rescale_result(image, bbox_contours, H, W):
    ori_H, ori_W = image.shape[:2]
    image = cv2.resize(image, (W, H))
    contours = list()
    for cont in bbox_contours:
        cont[:, 0] = (cont[:, 0] * W / ori_W).astype(int)
        cont[:, 1] = (cont[:, 1] * H / ori_H).astype(int)
        contours.append(cont)
    return image, contours

def norm2(x, axis=None):
    if axis:
        return np.sqrt(np.sum(x ** 2, axis=axis))
    return np.sqrt(np.sum(x ** 2))

def split_edge_seqence(points, n_parts):
    pts_num = points.shape[0]
    long_edge = [(i, (i + 1) % pts_num) for i in range(pts_num)]
    edge_length = [norm2(points[e1] - points[e2]) for e1, e2 in long_edge]
    point_cumsum = np.cumsum([0] + edge_length)
    total_length = sum(edge_length)
    length_per_part = total_length / n_parts

    cur_node = 0  # first point
    splited_result = []

    for i in range(1, n_parts):
        cur_end = i * length_per_part

        while cur_end > point_cumsum[cur_node + 1]:
            cur_node += 1

        e1, e2 = long_edge[cur_node]
        e1, e2 = points[e1], points[e2]

        # start_point = points[long_edge[cur_node]]
        end_shift = cur_end - point_cumsum[cur_node]
        ratio = end_shift / edge_length[cur_node]
        new_point = e1 + ratio * (e2 - e1)
        # print(cur_end, point_cumsum[cur_node], end_shift, edge_length[cur_node], '=', new_point)
        splited_result.append(new_point)

    # add first and last point
    p_first = points[long_edge[0][0]]
    p_last = points[long_edge[-1][1]]
    splited_result = [p_first] + splited_result + [p_last]
    return np.stack(splited_result)

def get_sample_point(text_mask, num_points, approx_factor, scales=None):
    # get sample point in contours
    contours, _ = cv2.findContours(text_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    epsilon = approx_factor * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True).reshape((-1, 2))
    # approx = contours[0].reshape((-1, 2))
    if scales is None:
        ctrl_points = split_edge_seqence(approx, num_points)
    else:
        ctrl_points = split_edge_seqence(approx*scales, num_points)
    ctrl_points = np.array(ctrl_points[:num_points, :]).astype(np.int32)

    return ctrl_points

def get_node_feature(cnn_feature, img_poly, ind, h, w):
    img_poly = img_poly.clone().float()
    img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
    img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1

    batch_size = cnn_feature.size(0)
    gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(img_poly.device)
    for i in range(batch_size):
        poly = img_poly[ind == i].unsqueeze(0)
        gcn_feature[ind == i] = torch.nn.functional.grid_sample(cnn_feature[i:i + 1], poly)[0].permute(1, 0, 2)
    return gcn_feature

def get_adj_ind(n_adj, n_nodes, device):
    ind = torch.tensor([i for i in range(-n_adj // 2, n_adj // 2 + 1) if i != 0]).long()
    ind = (torch.arange(n_nodes)[:, None] + ind[None]) % n_nodes
    return ind.to(device)

def get_adj_mat(n_adj, n_nodes):
    a = np.zeros([n_nodes, n_nodes], dtype=np.float)

    for i in range(n_nodes):
        for j in range(-n_adj // 2, n_adj // 2 + 1):
            if j != 0:
                a[i][(i + j) % n_nodes] = 1
                a[(i + j) % n_nodes][i] = 1
    return a

def normalize_adj(A, type="AD"):
    if type == "DAD":
        A = A + np.eye(A.shape[0])  # A=A+I
        d = np.sum(A, axis=0)
        d_inv = np.power(d, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_inv = np.diag(d_inv)
        G = A.dot(d_inv).transpose().dot(d_inv)  # L = D^-1/2 A D^-1/2
        G = torch.from_numpy(G)
    elif type == "AD":
        A = A + np.eye(A.shape[0])  # A=A+I
        A = torch.from_numpy(A)
        D = A.sum(1, keepdim=True)
        G = A.div(D)  # L= A/D
    else:
        A = A + np.eye(A.shape[0])  # A=A+I
        D = A.sum(1, keepdim=True)
        D = np.diag(D)
        G = torch.from_numpy(D - A)  # L = D-A
    return G

class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, features, A):
        x = torch.bmm(A, features)
        return x

class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, agg):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.FloatTensor(in_dim * 2, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        init.xavier_uniform_(self.weight)
        init.constant_(self.bias, 0)
        self.agg = agg()

    def forward(self, features, A):
        b, n, d = features.shape
        assert (d == self.in_dim)
        agg_feats = self.agg(features, A)
        cat_feats = torch.cat([features, agg_feats], dim=2)
        out = torch.einsum('bnd,df->bnf', (cat_feats, self.weight))
        out = F.relu(out + self.bias)
        return out

class GCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN, self).__init__()
        self.bn0 = nn.BatchNorm1d(in_dim, affine=False)

        self.conv1 = GraphConv(in_dim, 256, MeanAggregator)
        self.conv2 = GraphConv(256, 1024, MeanAggregator)
        self.conv3 = GraphConv(1024, 512, MeanAggregator)
        self.conv4 = GraphConv(512, out_dim, MeanAggregator)

        self.prediction = nn.Sequential(
            nn.Conv1d(out_dim, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2, 1))

    def forward(self, x, A):
        x = self.bn0(x)
        x = x.permute(0, 2, 1)
        b, n, c = x.shape
        A = A.expand(b, n, n)

        x = self.conv1(x, A)
        x = self.conv2(x, A)
        x = self.conv3(x, A)
        x = self.conv4(x, A)

        x = x.permute(0, 2, 1)
        pred = self.prediction(x)

        return pred

class RNN(nn.Module):
    def __init__(self, input, state_dim):
        super(RNN, self).__init__()
        self.bn0 = nn.BatchNorm1d(input, affine=False)
        self.rnn = nn.LSTM(input, state_dim, 1, dropout=0.1, bidirectional=True)
        self.prediction = nn.Sequential(
            nn.Conv1d(state_dim*2, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2, 1))

    def forward(self, x, adj):
        x = self.bn0(x)
        x = x.permute(2, 0, 1)
        x, _ = self.rnn(x)
        x = x.permute(1, 2, 0)
        pred = self.prediction(x)

        return pred
    
class CircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4):
        super(CircConv, self).__init__()

        self.n_adj = n_adj
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj*2+1)

    def forward(self, input, adj):
        input = torch.cat([input[..., -self.n_adj:], input, input[..., :self.n_adj]], dim=2)
        return self.fc(input)

class DilatedCircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4, dilation=1):
        super(DilatedCircConv, self).__init__()

        self.n_adj = n_adj
        self.dilation = dilation
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj*2+1, dilation=self.dilation)

    def forward(self, input, adj):
        if self.n_adj != 0:
            input = torch.cat([input[..., -self.n_adj*self.dilation:], input, input[..., :self.n_adj*self.dilation]], dim=2)
        return self.fc(input)

_conv_factory = {
    'grid': CircConv,
    'dgrid': DilatedCircConv
}

class BasicBlock(nn.Module):
    def __init__(self, state_dim, out_state_dim, conv_type, n_adj=4, dilation=1):
        super(BasicBlock, self).__init__()

        self.conv = _conv_factory[conv_type](state_dim, out_state_dim, n_adj, dilation)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm1d(out_state_dim)

    def forward(self, x, adj=None):
        x = self.conv(x, adj)
        x = self.relu(x)
        x = self.norm(x)
        return x

class DeepSnake(nn.Module):
    def __init__(self, state_dim, feature_dim, conv_type='dgrid'):
        super(DeepSnake, self).__init__()

        self.head = BasicBlock(feature_dim, state_dim, conv_type)

        self.res_layer_num = 7
        dilation = [1, 1, 1, 2, 2, 4, 4]
        for i in range(self.res_layer_num):
            conv = BasicBlock(state_dim, state_dim, conv_type, n_adj=4, dilation=dilation[i])
            self.__setattr__('res'+str(i), conv)

        fusion_state_dim = 256
        self.fusion = nn.Conv1d(state_dim * (self.res_layer_num + 1), fusion_state_dim, 1)
        self.prediction = nn.Sequential(
            nn.Conv1d(state_dim * (self.res_layer_num + 1) + fusion_state_dim, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2, 1)
        )

    def forward(self, x, adj):
        states = []

        x = self.head(x, adj)
        states.append(x)
        for i in range(self.res_layer_num):
            x = self.__getattr__('res'+str(i))(x, adj) + x
            states.append(x)

        state = torch.cat(states, dim=1)
        global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]
        global_state = global_state.expand(global_state.size(0), global_state.size(1), state.size(2))
        state = torch.cat([global_state, state], dim=1)
        x = self.prediction(state)

        return x

class AdaptiveDeformation(nn.Module):
    def __init__(self, input, state_dim):
        super(AdaptiveDeformation, self).__init__()
        self.bn0 = nn.BatchNorm1d(input, affine=False)
        self.conv1 = nn.Conv1d(input, state_dim, 1)
        self.rnn = nn.LSTM(input, state_dim, 1, bidirectional=True)
        self.gconv1 = GraphConv(input, 256, MeanAggregator)
        self.gconv2 = GraphConv(256, 1024, MeanAggregator)
        self.gconv3 = GraphConv(1024, 512, MeanAggregator)
        self.gconv4 = GraphConv(512, state_dim, MeanAggregator)

        self.prediction = nn.Sequential(
            nn.Conv1d(4*state_dim, 128, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(64, 2, 1))

    def forward(self, x, A):
        x = self.bn0(x)

        # # rnn block
        yl = x.permute(2, 0, 1)
        yl, _ = self.rnn(yl)
        yl = yl.permute(1, 2, 0)

        # # gcn block
        yg = x.permute(0, 2, 1)
        b, n, c = yg.shape
        A = A.expand(b, n, n)
        yg = self.gconv1(yg, A)
        yg = self.gconv2(yg, A)
        yg = self.gconv3(yg, A)
        yg = self.gconv4(yg, A)
        yg = yg.permute(0, 2, 1)

        # res block
        x = torch.cat([yl, yg, self.conv1(x)], dim=1)
        pred = self.prediction(x)

        return pred

class Evolution(nn.Module):
    def __init__(self, node_num, adj_num, is_training=False, device=None, model="snake"):
        super(Evolution, self).__init__()
        self.node_num = node_num
        self.adj_num = adj_num
        self.device = device
        self.is_training = is_training
        self.clip_dis = 16

        self.iter = 3
        if model == "gcn":
            self.adj = get_adj_mat(self.adj_num, self.node_num)
            self.adj = normalize_adj(self.adj, type="DAD").float().to(self.device)
            for i in range(self.iter):
                evolve_gcn = GCN(36, 128)
                self.__setattr__('evolve_gcn' + str(i), evolve_gcn)
        elif model == "rnn":
            self.adj = None
            for i in range(self.iter):
                evolve_gcn = RNN(36, 128)
                self.__setattr__('evolve_gcn' + str(i), evolve_gcn)
        elif model == "AD":
            self.adj = get_adj_mat(self.adj_num, self.node_num)
            self.adj = normalize_adj(self.adj, type="DAD").float().to(self.device)
            for i in range(self.iter):
                evolve_gcn = AdaptiveDeformation(36, 128)
                self.__setattr__('evolve_gcn' + str(i), evolve_gcn)
        elif model == "BT":
            self.adj = None
            for i in range(self.iter):
                evolve_gcn = Transformer(36, 128, num_heads=8,
                                         dim_feedforward=1024, drop_rate=0.0, if_resi=True, block_nums=3)
                self.__setattr__('evolve_gcn' + str(i), evolve_gcn)
        else:
            self.adj = get_adj_ind(self.adj_num, self.node_num, self.device)
            for i in range(self.iter):
                evolve_gcn = DeepSnake(state_dim=128, feature_dim=36, conv_type='dgrid')
                self.__setattr__('evolve_gcn' + str(i), evolve_gcn)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                # nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @staticmethod
    def get_boundary_proposal(input=None, seg_preds=None, switch="gt"):

        if switch == "gt":
            inds = torch.where(input['ignore_tags'] > 0)
            # if len(inds[0]) > 320:
            #    inds = (inds[0][:320], inds[1][:320])
            init_polys = input['proposal_points'][inds]
        else:
            tr_masks = input['tr_mask'].cpu().numpy()
            tcl_masks = seg_preds[:, 0, :, :].detach().cpu().numpy() > 0.5
            inds = []
            init_polys = []
            for bid, tcl_mask in enumerate(tcl_masks):
                ret, labels = cv2.connectedComponents(tcl_mask.astype(np.uint8), connectivity=8)
                for idx in range(1, ret):
                    text_mask = labels == idx
                    ist_id = int(np.sum(text_mask*tr_masks[bid])/np.sum(text_mask))-1
                    inds.append([bid, ist_id])
                    poly = get_sample_point(text_mask, 20, 0.004)
                    init_polys.append(poly)
            inds = torch.from_numpy(np.array(inds)).permute(1, 0).to(input["img"].device)
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device)

        return init_polys, inds, None

    def get_boundary_proposal_eval(self, input=None, seg_preds=None):

        # if 1 > 1:
        #     seg_preds = F.interpolate(seg_preds, scale_factor=1, mode='bilinear')
        cls_preds = seg_preds[:, 0, :, :].detach().cpu().numpy()
        dis_preds = seg_preds[:, 1, :, ].detach().cpu().numpy()

        inds = []
        init_polys = []
        confidences = []
        for bid, dis_pred in enumerate(dis_preds):
            # # dis_mask = (dis_pred / np.max(dis_pred)) > 0.35
            dis_mask = dis_pred > 0.35
            # dis_mask = fill_hole(dis_mask)
            ret, labels = cv2.connectedComponents(dis_mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_16U)
            for idx in range(1, ret):
                text_mask = labels == idx
                confidence = round(cls_preds[bid][text_mask].mean(), 3)
                # 50 for MLT2017 and ArT (or DCN is used in backone); else is all 150;
                # just can set to 50, which has little effect on the performance
                if np.sum(text_mask) < 50/(1*1) or confidence < 0.9:
                    continue
                confidences.append(confidence)
                inds.append([bid, 0])
                
                poly = get_sample_point(text_mask, 20,
                                        0.004, scales=np.array([1, 1]))
                init_polys.append(poly)

        if len(inds) > 0:
            inds = torch.from_numpy(np.array(inds)).permute(1, 0).to(input["img"].device, non_blocking=True)
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device, non_blocking=True).float()
        else:
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device, non_blocking=True).float()
            inds = torch.from_numpy(np.array(inds)).to(input["img"].device, non_blocking=True)

        return init_polys, inds, confidences

    def evolve_poly(self, snake, cnn_feature, i_it_poly, ind):
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2)*1, cnn_feature.size(3)*1
        node_feats = get_node_feature(cnn_feature, i_it_poly, ind, h, w)
        tmp = snake(node_feats)
        tmp = tmp.permute(0, 2, 1)
        i_poly = i_it_poly + torch.clamp(tmp, -self.clip_dis, self.clip_dis)
        if self.is_training:
            i_poly = torch.clamp(i_poly, 0, w-1)
        else:
            i_poly[:, :, 0] = torch.clamp(i_poly[:, :, 0], 0, w - 1)
            i_poly[:, :, 1] = torch.clamp(i_poly[:, :, 1], 0, h - 1)
        return i_poly

    def forward(self, embed_feature, input=None, seg_preds=None, switch="gt"):
        if self.is_training:
            init_polys, inds, confidences = self.get_boundary_proposal(input=input, seg_preds=seg_preds, switch=switch)
            # TODO sample fix number
        else:
            init_polys, inds, confidences = self.get_boundary_proposal_eval(input=input, seg_preds=seg_preds)
            if init_polys.shape[0] == 0:
                return [init_polys for i in range(self.iter+1)], inds, confidences

        py_preds = [init_polys, ]
        for i in range(self.iter):
            evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
            init_polys = self.evolve_poly(evolve_gcn, embed_feature, init_polys, inds[0])
            py_preds.append(init_polys)

        return py_preds, inds, confidences

class UpBlok(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, upsampled, shortcut):
        x = torch.cat([upsampled, shortcut], dim=1)
        x = self.conv1x1(x)
        x = F.relu(x)
        x = self.conv3x3(x)
        x = F.relu(x)
        x = self.deconv(x)
        return x

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        base_net = resnet.resnet50(pretrained=False)

        self.stage1 = nn.Sequential(
            base_net.conv1,
            base_net.bn1,
            base_net.relu,
            base_net.maxpool
        )
        self.stage2 = base_net.layer1
        self.stage3 = base_net.layer2
        self.stage4 = base_net.layer3
        self.stage5 = base_net.layer4
        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        C1 = self.stage1(x)
        C2 = self.stage2(C1)
        C3 = self.stage3(C2)
        C4 = self.stage4(C3)
        C5 = self.stage5(C4)
        C1 = self.up2(C1)

        return C1, C2, C3, C4, C5

class FPN(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = ResNet()
        self.deconv5 = nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1)
        self.merge4 = UpBlok(1024 + 256, 128)
        self.merge3 = UpBlok(512 + 128, 64)
        self.merge2 = UpBlok(256 + 64, 32)  # FPN 1/2
        self.merge1 = UpBlok(64 + 32, 32)   # FPN 1/1

    def forward(self, x):
        C1, C2, C3, C4, C5 = self.backbone(x)
        up5 = self.deconv5(C5)
        up5 = F.relu(up5)

        up4 = self.merge4(C4, up5)
        up4 = F.relu(up4)

        up3 = self.merge3(C3, up4)
        up3 = F.relu(up3)

        up2 = self.merge2(C2, up3)
        up2 = F.relu(up2)

        up1 = self.merge1(C1, up2)
        up1 = F.relu(up1)

        return up1, up2, up3, up4, up5

class TextNet(nn.Module):

    def __init__(self, device, is_training=False):
        super().__init__()
        self.device = device
        self.is_training = is_training
        self.fpn = FPN()

        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=2, dilation=2),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=4, dilation=4),
            nn.PReLU(),
            nn.Conv2d(16, 4, kernel_size=1, stride=1, padding=0),
        )
        self.BPN = Evolution(20, adj_num=4, is_training=is_training, device=self.device, model="BT")

    def load_model(self, model_path):
        print('Loading from {}'.format(model_path))
        state_dict = torch.load(model_path, map_location=torch.device(self.device))
        self.load_state_dict(state_dict['model'], strict=(not self.is_training))

    def forward(self, input_dict, test_speed=False):
        output = {}
        b, c, h, w = input_dict["img"].shape
        if self.is_training or "Totaltext" in ['ArT', 'MLT2017', "MLT2019"] or test_speed:
            image = input_dict["img"]
        else:
            image = torch.zeros((b, c, 1024, 1024), dtype=torch.float32).to(self.device)
            image[:, :, :h, :w] = input_dict["img"][:, :, :, :]

        up1, _, _, _, _ = self.fpn(image)
        up1 = up1[:, :, :h // 1, :w // 1]

        preds = self.seg_head(up1)
        fy_preds = torch.cat([torch.sigmoid(preds[:, 0:2, :, :]), preds[:, 2:4, :, :]], dim=1)
        cnn_feats = torch.cat([up1, fy_preds], dim=1)

        py_preds, inds, confidences = self.BPN(cnn_feats, input=input_dict, seg_preds=fy_preds, switch="gt")
        
        output["fy_preds"] = fy_preds
        output["py_preds"] = py_preds
        output["inds"] = inds
        output["confidences"] = confidences

        return output

def inference(model, imgs, transform, conf):
    art_results = dict()
    for i, (image, meta) in enumerate(imgs):
        input_dict = dict()
        idx = 0  # test mode can only run with batch_size == 1
        H, W = meta['Height'][idx].item(), meta['Width'][idx].item()
 
        image = transform(image)
        input_dict['img'] = image.to(conf["device"])

        # get detection result
        output_dict = model(input_dict)
        torch.cuda.synchronize()

        # visualization
        img_show = image[idx].permute(1, 2, 0).cpu().numpy()
        img_show = ((img_show * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255).astype(np.uint8)

        if conf["contour"]["save_results"]:
            gt_contour = []
            label_tag = meta['label_tag'][idx].int().cpu().numpy()
            for annot, n_annot in zip(meta['annotation'][idx], meta['n_annotation'][idx]):
                if n_annot.item() > 0:
                    gt_contour.append(annot[:n_annot].int().cpu().numpy())

            gt_vis = visualize_gt(img_show, gt_contour, label_tag)
            show_boundary, heat_map = visualize_detection(img_show, conf["save_dir"], output_dict, meta=meta)

            show_map = np.concatenate([heat_map, gt_vis], axis=1)
            show_map = cv2.resize(show_map, (320 * 3, 320))
            im_vis = np.concatenate([show_map, show_boundary], axis=0)

            path = os.path.join(conf["save_dir"], "contour_vis.jpg")
            cv2.imwrite(path, im_vis)

        contours = output_dict["py_preds"][-1].int().cpu().numpy()
        img_show, contours = rescale_result(img_show, contours, H, W)
        
        for cont in contours:
            cont = np.stack([cont[:, 0], cont[:, 1]], 1)
            if cv2.contourArea(cont) <= 0:
                continue
            cont = cont.flatten().astype(str).tolist()
            cont = ','.join(cont)

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, pts=None):
        for t in self.transforms:
            img, pts = t(img, pts)
        return img, pts

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image, polygons=None):
        image = image.astype(np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.std
        return image, polygons

class ResizeSquare(object):
    def __init__(self, size=(480, 1280)):
        self.size = size

    def __call__(self, image, polygons=None):
        h, w, _ = image.shape
        img_size_min = min(h, w)
        img_size_max = max(h, w)

        if img_size_min < self.size[0]:
            im_scale = float(self.size[0]) / float(img_size_min)  # expand min to size[0]
            if np.ceil(im_scale * img_size_max) > self.size[1]:  # expand max can't > size[1]
                im_scale = float(self.size[1]) / float(img_size_max)
        elif img_size_max > self.size[1]:
            im_scale = float(self.size[1]) / float(img_size_max)
        else:
            im_scale = 1.0

        new_h = int(int(h * im_scale/32)*32)
        new_w = int(int(w * im_scale/32)*32)
        image = cv2.resize(image, (new_w, new_h))
        scales = np.array([new_w / w, new_h / h])
        if polygons is not None:
            for polygon in polygons:
                polygon.points = polygon.points * scales

        return image, polygons
    
class BaseTransform(object):
    def __init__(self, size, mean, std):
        self.size = size
        self.mean = mean
        self.std = std
        self.augmentation = Compose([
            ResizeSquare(size=self.size),
            Normalize(mean, std)
        ])

    def __call__(self, image, polygons=None):
        return self.augmentation(image, polygons)

def load_ctr_model(gpu_id, model_path):
    device = f"cuda:{gpu_id}"
    model = TextNet(device=device, is_training=False)
    model.load_model(model_path)
    model = model.to(device)
    model.eval()
    return model

def sort_contours(contours):
    def avg_second_element(sub_array):
        return np.mean(sub_array[:, 1])
    contours = sorted(contours, key=avg_second_element)
    return list(contours)

def order_text(union):
    sorted_union = sorted(union, key=lambda item: item[1][0])
    ordered = " ".join(item[0] for item in sorted_union)
    scores = [item[3] for item in sorted_union]
    return {"sorted_union":sorted_union, "ordered_text":ordered, "scores":scores}

def get_banner_id_set(str_results):
    result = []
    for str_result in str_results:
        for str_r in str_result:
            result.append(str_r[2])
    return set(result)

def merge_texts(b_ids, unions):
    texts = {}
    for b_id in b_ids:
        text = ''
        scores = []
        for u in unions:
            if u['banner_id'] == b_id:
                text = " ".join([text, u['ordered_text']]).strip()
                scores.extend(u["scores"])
        texts[b_id] = {"ori_txt":text, "score":scores}
    return texts

def union_text(str_results, contours):
    result = []
    b_ids = get_banner_id_set(str_results)
    for contour in contours:
        unions = []
        contour = sort_contours(contour)
        for b_id, str_result in enumerate(str_results):
            
            for cont in contour:
                union = []
                
                for str_r in str_result:
                    cxcy = str_r[1]
                    path = mpath.Path(cont)
                    is_inside = path.contains_point(cxcy)
                    if is_inside: union.append(str_r)
                union_item = order_text(union)
                union_item["banner_id"] = b_id
                if union_item["ordered_text"]:
                    unions.append(union_item)
            
        texts = merge_texts(b_ids, unions)
        result.append(texts)
    return result

def union_text_rev(str_results, contours):
    result = []
    b_ids = get_banner_id_set(str_results)
    unions = []
    for b_id, str_result in enumerate(str_results): 
        union = []
        for str_r in str_result: 
            union.append(str_r) 
        union_item = order_text(union) 
        union_item["banner_id"] = b_id 
        if union_item["ordered_text"]:
            unions.append(union_item) 
    texts = merge_texts(b_ids, unions)
    result.append(texts)
    return result 


class KoBARTSummaryDataset(Dataset):
    def __init__(self, file, tokenizer, max_len, ignore_index=-100):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.docs = pd.read_csv(file, sep='\t')
        self.len = self.docs.shape[0]

        self.pad_index = self.tokenizer.pad_token_id
        self.ignore_index = ignore_index

        self.class_map = {
            '정당': 0,
            '공공': 1,
            '민간': 2,
            '알 수 없음': 3
        }

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] *(self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] *(self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs
    
    def __getitem__(self, idx):
        instance = self.docs.iloc[idx]
        input_ids = self.tokenizer.encode(str(instance['original']))
        input_ids = self.add_padding_data(input_ids)

        label_ids = self.tokenizer.encode(str(instance['refined']))
        label_ids.append(self.tokenizer.eos_token_id)
        dec_input_ids = [self.tokenizer.eos_token_id]
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self.add_padding_data(dec_input_ids)
        label_ids = self.add_ignored_data(label_ids)

        class_ids = self.class_map[instance['class']]
        
        return {
            'input_ids': np.array(input_ids, dtype=np.int64),  # np.int64로 명시적 변환
            'decoder_input_ids': np.array(dec_input_ids, dtype=np.int64),  # np.int64로 명시적 변환
            'labels': np.array(label_ids, dtype=np.int64),  # np.int64로 명시적 변환
            'class': np.array(class_ids, dtype=np.int64)  # np.int64로 명시적 변환
        }

    def __len__(self):
        return self.len

class KobartSummaryModule(L.LightningDataModule):
    def __init__(self, train_file,
                 test_file, tok,
                 max_len=512,
                 batch_size=8,
                 num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.max_len = max_len
        self.train_file_path = train_file
        self.test_file_path = test_file
        self.tok = tok
        self.num_workers = num_workers

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers',
                            type=int,
                            default=1,
                            help='num of worker for dataloader')
        return parser

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        self.train = KoBARTSummaryDataset(self.train_file_path,
                                 self.tok,
                                 self.max_len)
        self.test = KoBARTSummaryDataset(self.test_file_path,
                                self.tok,
                                self.max_len)

    def train_dataloader(self):
        train = DataLoader(self.train,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers, shuffle=True)
        return train

    def val_dataloader(self):
        val = DataLoader(self.test,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers, shuffle=False)
        return val

    def test_dataloader(self):
        test = DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)
        return test

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),  # BatchNorm added
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),  # BatchNorm added
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),  # BatchNorm added
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),  # BatchNorm added
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),  # BatchNorm added
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),  # BatchNorm added
            nn.ReLU(),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),  # BatchNorm added
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),  # BatchNorm added
            nn.ReLU(),
        )
        
        # 최종적으로 결합된 feature를 사용하는 FC layer
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4, 256),
            nn.BatchNorm1d(256),  # BatchNorm added
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, hidden_states):
        # 각각의 hidden states에 대해 독립적으로 처리
        encoded1 = self.fc1(hidden_states[-1].mean(dim=1))
        encoded3 = self.fc2(hidden_states[-3].mean(dim=1))
        encoded5 = self.fc3(hidden_states[-5].mean(dim=1))
        encoded7 = self.fc4(hidden_states[-7].mean(dim=1))

        # concat하여 최종 classifier에 입력
        combined_features = torch.cat((encoded1, encoded3, encoded5, encoded7), dim=-1)  # [batch_size, 256 * 3]
        logits = self.classifier(combined_features)  # [batch_size, num_classes]
        
        return logits

class KoBARTConditionalGeneration(L.LightningModule):
    def __init__(
        self,
        hparams,
        **kwargs):
        
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v1')
        self.model.train()
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
        self.pad_token_id = self.tokenizer.pad_token_id

        hidden_dim = self.model.config.d_model
        self.cls_head = ClassificationHead(input_dim=hidden_dim, num_classes=hparams.num_classes)
        self.ce = nn.CrossEntropyLoss()
        
        self.outputs = defaultdict(list)
            
    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters()) + list(self.cls_head.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        num_workers = self.hparams.num_workers

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.trainer.estimated_stepping_batches * 0.1),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        lr_scheduler = {'scheduler': scheduler, 
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]
    
    def forward(self, inputs):
        attention_mask = inputs['input_ids'].ne(self.pad_token_id).float()
        decoder_attention_mask = inputs['decoder_input_ids'].ne(self.pad_token_id).float()
        
        # 모델의 출력을 가져옵니다.
        outputs = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=attention_mask,
            decoder_input_ids=inputs['decoder_input_ids'],
            decoder_attention_mask=decoder_attention_mask,
            labels=inputs['labels'], 
            return_dict=True,
            output_hidden_states=True,  # hidden states 반환
            output_attentions=True  # attention weights 반환
        )

        hidden_states = outputs.decoder_hidden_states

        logits = self.cls_head(hidden_states)
        
        return outputs, logits, outputs.decoder_attentions
    
    def training_step(self, batch, batch_idx):
        outs, logits, _ = self(batch)
        reg_loss = outs.loss
        cls_loss = self.ce(logits, batch['class'])
        loss = reg_loss + cls_loss
        self.log('train_reg_loss', reg_loss)
        self.log('train_cls_loss', cls_loss)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outs, logits, attentions = self(batch)
        reg_loss = outs['loss']
        cls_loss = self.ce(logits, batch['class'])
        loss = reg_loss + cls_loss
    
        self.outputs[dataloader_idx].append({"reg_loss": reg_loss, 
                                             "cls_loss": cls_loss, 
                                             "loss": loss,
                                             })

    def on_validation_epoch_end(self):
        flat_outputs = []
        for lst in self.outputs.values():
            flat_outputs.extend(lst)
        reg_loss = torch.stack([x["reg_loss"] for x in flat_outputs]).mean()
        cls_loss = torch.stack([x["cls_loss"] for x in flat_outputs]).mean()
        loss = torch.stack([x["loss"] for x in flat_outputs]).mean()
        self.log("val_reg_loss", reg_loss)
        self.log("val_cls_loss", cls_loss)
        self.log("val_loss", loss, prog_bar=True)
        self.outputs.clear()

class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--train_file',
                            type=str,
                            default='data/refine/train.tsv',
                            help='train file')
        parser.add_argument('--test_file',
                            type=str,
                            default='data/refine/test.tsv',
                            help='test file')
        parser.add_argument('--batch_size',
                            type=int,
                            default=8,
                            help='')
        parser.add_argument('--checkpoint',
                            type=str,
                            default='checkpoint',
                            help='')
        parser.add_argument('--max_len',
                            type=int,
                            default=512,
                            help='max seq len')
        parser.add_argument('--max_epochs',
                            type=int,
                            default=10,
                            help='train epochs')
        parser.add_argument('--lr',
                            type=float,
                            default=3e-5,
                            help='The initial learning rate')
        parser.add_argument('--accelerator',
                            type=str,
                            default='gpu',
                            choices=['gpu', 'cpu'],
                            help='select accelerator')
        parser.add_argument('--num_gpus',
                            type=int,
                            default=1,
                            help='number of gpus')
        parser.add_argument('--gpu_id',
                            type=int,
                            default=0,
                            nargs='+',
                            help='list of gpu ids')
        parser.add_argument('--num_classes',
                            type=int,
                            default=4,
                            help='number of classes')
        parser.add_argument('--gradient_clip_val',
                            type=float,
                            default=1.0,
                            help='gradient_clipping')
        parser.add_argument("--img_name", type=str, default='cctv_samples_small/1.jpg')
        return parser

def load_ref_cls_model(gpu_id, model_path):
    parser = argparse.ArgumentParser(description='KoBART Refinement')
    parser = ArgsBase.add_model_specific_args(parser)
    parser = KobartSummaryModule.add_model_specific_args(parser)
    args = parser.parse_args()

    model = KoBARTConditionalGeneration.load_from_checkpoint(model_path, 
                                                             hparams=args,
                                                             map_location=f"cuda:{gpu_id}")
    model.eval()
    return model, args

def preprocess_text(tokenizer, text_batch, device, max_len=512):
    # 입력 텍스트를 인코딩하고, 최대 길이만큼 패딩을 추가
    input_ids = []
    dec_input_ids = []
    for text in text_batch:
        input_id = tokenizer.encode(text)
        if len(input_id) < max_len:
            pad = [tokenizer.pad_token_id] * (max_len - len(input_id))
            input_id = input_id + pad
        else:
            input_id = input_id[:max_len]
        
        input_ids.append(torch.tensor(input_id))
            
        # 디코더 입력을 위해 <eos> 토큰 추가 및 패딩 처리
        dec_input_id = [tokenizer.eos_token_id] + input_id[:-1]
        if len(dec_input_id) < max_len:
            pad = [tokenizer.pad_token_id] * (max_len - len(dec_input_id))
            dec_input_id = dec_input_id + pad
        else:
            dec_input_id = dec_input_id[:max_len]
        
        dec_input_ids.append(torch.tensor(dec_input_id))
    
    input_ids = torch.stack(input_ids)
    dec_input_ids = torch.stack(dec_input_ids)
        
    # 모델에 입력 데이터를 전달하고 예측 수행
    inputs = {
        'input_ids': input_ids.to(device),
        'decoder_input_ids': dec_input_ids.to(device),
        'labels': None
    }
    return inputs

def banRefAndCls(model, device, ordered_text):
    classes = ['정당','공공','민간','알 수 없음']

    results = []
    for ot in ordered_text:
        b_ids = list(map(int, ot.keys()))
        text_batch = [item["ori_txt"] for item in ot.values()]
        
        inputs = preprocess_text(model.tokenizer, text_batch, device)

        generated_ids = model.model.generate(
            inputs["input_ids"],
            max_length=50,  # 최대 길이 제한
            num_beams=5,    # 빔 서치
            early_stopping=True,  # <eos> 토큰에서 중지
            no_repeat_ngram_size=2  # 반복 방지
        )
        _, logits, _ = model(inputs)

        generated_texts = [model.tokenizer.decode(gen_id, skip_special_tokens=True) for gen_id in generated_ids]
        predicted_classes = [classes[p_idx] for p_idx in torch.argmax(logits, dim=-1)]
        
        result = []
        for idx in range(len(b_ids)):
            result.append({"banner_id":b_ids[idx],
                            "refined_text":generated_texts[idx],
                            "class":predicted_classes[idx]
                         })
        results.append(result)

    return results

def banContour(model, imgs, conf):

    save_dir = os.path.join(conf.save_root_path, "contour")
    conf["save_dir"] = save_dir
    conf["device"] = model.device

    with torch.no_grad():
        imgs_contours = list()
        for image in imgs:
            _, H, W = image.shape
            points = np.zeros((64, 20, 2))
            length = np.zeros(64, dtype=int)
            label_tag = np.zeros(64, dtype=int)

            meta = {
                'annotation': points,
                'n_annotation': length,
                'label_tag': label_tag,
                'Height': H,
                'Width': W
            }

            input_dict = dict()
            input_dict['img'] = image.unsqueeze(0).to(conf["device"])

            # get detection result
            output_dict = model(input_dict)
            torch.cuda.synchronize()

            # visualization
            img_show = image.permute(1, 2, 0).cpu().numpy()
            img_show = ((img_show * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255).astype(np.uint8)

            if conf["contour"]["save_results"]:
                os.makedirs(save_dir)
                gt_contour = []
                label_tag = meta['label_tag']
                for annot, n_annot in zip(meta['annotation'], meta['n_annotation']):
                    if n_annot.item() > 0:
                        gt_contour.append(annot[:n_annot])

                gt_vis = visualize_gt(img_show, gt_contour, label_tag)
                show_boundary, heat_map = visualize_detection(img_show, conf["save_dir"], output_dict, meta=meta)

                show_map = np.concatenate([heat_map, gt_vis], axis=1)
                show_map = cv2.resize(show_map, (320 * 3, 320))
                im_vis = np.concatenate([show_map, show_boundary], axis=0)

                path = os.path.join(conf["save_dir"], "contour_vis.jpg")
                cv2.imwrite(path, im_vis)

            contours = output_dict["py_preds"][-1].cpu().detach().numpy().astype(np.int64)
            img_show, contours = rescale_result(img_show, contours, H, W)
            imgs_contours.append(contours)
        return imgs_contours

def banDet(det_model, test_pipeline, imgs_array, imgs, params, data):
    print("Step 1: Banner Detection running...")
    
    result = inference_detector(det_model, imgs_array, test_pipeline=test_pipeline, text_prompt='outdoor banner')[0].pred_instances # 'license plate'
    
    bboxes = result.bboxes
    scores = result.scores.unsqueeze(-1)
    labels = result.labels.unsqueeze(-1)
    
    inf_out_merged = torch.cat([bboxes,scores,labels], dim=-1)
    if len(inf_out_merged.shape) == 2:
        inf_out_merged = inf_out_merged.unsqueeze(0)

    output_merged = inf_out_merged[inf_out_merged[:,:,4]>params.detection.conf_thres]
    
    if len(output_merged.shape) == 2:
        output_merged = output_merged.unsqueeze(0)

    print(output_merged.shape)
    
    if params.detection.save_results:
        os.makedirs(os.path.join(params.save_root_path), exist_ok=True)
        for i, (img, output) in enumerate(zip(imgs, output_merged)):
            img = img.permute(1, 2, 0).detach().cpu().numpy() * 255
            img = cv2.resize(img, (data[i]['width'], data[i]['height']))
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
            if output is not None:
                for *xyxy, conf, _ in output.tolist():
                    x1, y1, x2, y2 = xyxy
                    try:
                        bb.add(img, x1/1024*data[i]['width'], y1/1024*data[i]['height'], x2/1024*data[i]['width'], y2/1024*data[i]['height'], format(f"{conf:.2f}"))
                    except:
                        pass
            filename = params.out_img_filename.split(".")
            filename[0] += f"_det{i}"
            filename = ".".join(filename)
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(
                params.save_root_path, filename), img)

    return output_merged

def banSeg(sam_model, imgs, img_points, image_idx, banner_idx, params, device='cuda'):
    seg_out = []
    scores = []
    
    total_t = time.time()
    infer_t = time.time()
    out = sam_model.inference_step(imgs, img_points)
    pred_masks = out['predictions']
    confidence_predictions = out['confidence_predictions']
    # print(f"inference time: {time.time() - infer_t}")

    loop_t = time.time()
    for i, (pred_mask) in enumerate(pred_masks):

        pre_t = time.time()
        pred_mask = pred_mask > params.segmentation.mask_threshold

        banner_idx[0] -= 1
        # banner_idx[image_idx] -= 1
        # np.add.at(banner_idx, image_idx, -1)

        score = confidence_predictions[i].detach().cpu().numpy()[0][0]
        # print(f"preprocessing time: {time.time() - pre_t}")

        save_t = time.time()
        if params.segmentation.save_results:
            os.makedirs(os.path.join(
                params.save_root_path), exist_ok=True)
            filename = params.out_img_filename.split(".")
            filename[0] += "_seg"
            filename = ".".join(filename)

            img = imgs[i].permute(1, 2, 0).detach().cpu().numpy() * 255
            point = img_points[i].detach().cpu().numpy()
            point_label = torch.ones(
                (point.shape[0],), dtype=torch.int, device=device)
            plt.figure(figsize=(10, 10))
            img_plot = cv2.cvtColor(
                img.astype(np.uint8), cv2.COLOR_RGB2BGR)
            img_plot = cv2.cvtColor(
                img_plot.astype(np.uint8), cv2.COLOR_BGR2RGB)
            plt.imshow(img_plot)
            show_mask(pred_mask, plt.gca())
            show_points(
                point, point_label.detach().cpu().numpy(), plt.gca())
            plt.title(f"Confidence: {score:.3f}", fontsize=12)
            plt.axis('off')
            plt.savefig(os.path.join(params.save_root_path, filename),
                        bbox_inches='tight', pad_inches=0)
            plt.close()
        # print(f"save time: {time.time() - save_t}")

        post_t = time.time()
        seg_out.append(pred_mask.squeeze(0).squeeze(
            0).detach().cpu().numpy().astype(np.uint8))

        scores.append(score)
        # print(f"postprocessing time: {time.time() - post_t}")
    # print(f"loop time: {time.time() - loop_t}")
    # print(f"total processing time: {time.time() - total_t}")
    # print("="*50)

    return seg_out, scores

def banSTD(std_model, imgs, args, gpu_id=None):
    print("Step 2: Scene Text Detection running...")
    std_outs = []
    poly_outs = []
    for img in imgs:
        # CRAFT
        bbox, polys, score_text = test_net(std_model, img, args.scene_text_detection.text_threshold,
                                           args.scene_text_detection.link_threshold, args.scene_text_detection.low_text, args.scene_text_detection.poly, args, gpu_id=gpu_id)

        if len(polys) == 0:
            std_outs.append([])
            poly_outs.append([])
            continue

        std_out = []
        poly_out = []

        for i in range(len(polys)):
            min_width = int(polys[i][:, 0].min())
            max_width = int(polys[i][:, 0].max())
            margin_width = int((max_width - min_width) * 0)
            min_width = max(0, min_width - margin_width)
            max_width = min(img.shape[1], max_width + margin_width)
            min_height = int(polys[i][:, 1].min())
            max_height = int(polys[i][:, 1].max())
            margin_height = int((max_height - min_height) * 0)
            min_height = max(0, min_height - margin_height)
            max_height = min(img.shape[0], max_height + margin_height)

            std_out.append(
                img[min_height:max_height, min_width:max_width])

            poly_out.append([min_width, min_height,
                             max_width, max_height])

        std_outs.append(std_out)
        poly_outs.append(poly_out)

    return std_outs, poly_outs

def extract_nouns(text):
    mecab = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ko-dic")
    parsed = mecab.parse(text)
    nouns = []
    
    for line in parsed.splitlines():
        if '\t' in line:
            word, feature = line.split('\t')
            features = feature.split(',')
            if features[0] == 'NNG' or features[0] == 'NNP':  # 일반 명사와 고유 명사
                nouns.append(word)
    
    return nouns

def crop_n_resize(imgs, det_points):
    indices = []
    cropped_imgs = []
    four_corners = []
    for idx in range(imgs.shape[0]):
        points = det_points[idx]
        for p_idx in range(points.shape[0]):
            x1, y1, x2, y2 = map(lambda x: 0 if x < 0 else x, map(int, points[p_idx, :4].tolist()))
            cropped_img = imgs[idx][:, y1:y2, x1:x2]

            cropped_img = cropped_img.permute(1, 2, 0).cpu().numpy()
            cropped_img = np.clip(cropped_img * 255, 0, 255)
            cropped_imgs.append(cropped_img)
            indices.append(idx)
            four_corners.append([[x1,y1], [x1,y2], [x2,y1], [x2,y2]])

    return cropped_imgs, indices, four_corners

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Banner:
    def __init__(self, gpu_id):
        self.gpu_ids = {
            "det":gpu_id,
            "sam":gpu_id,
            "std":gpu_id,
            "str":gpu_id,
            "ctr":gpu_id,
            "ref_cls":gpu_id}

        self.str_preprocess = transforms.Compose(
            [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        self.resize = Resize((32, 128))

    @torch.no_grad()
    def inference(self, cctv_id, path, models, params):
        (self.det_model, self.test_pipeline), self.std_model, self.str_model, self.ctr_model, self.ref_cls_model = models

        start_time = time.time()

        save_root_path = os.path.join(
            params.save_root_path, params.save_path)
        params.save_root_path = save_root_path
        os.makedirs(save_root_path, exist_ok=True)

        if params.verbose == True:
            logging.basicConfig(filename=os.path.join(save_root_path, "inference.log"),
                                filemode="a", format='%(asctime)s - %(message)s', level=logging.INFO, force=True)
            logging.info("Start time : {}".format(datetime.now()))
            logging.info("Params : {}".format(params))

        # Preprocess
        imgs, ratio, original_size, imgs_array = preprocess_image(
            path, params, f"cuda:{self.gpu_ids['det']}")

        if params.verbose == True:
            logging.info("Preprocess time : {}".format(
                time.time() - start_time))
            preprocessing_time = time.time()
            logging.info("Image shape : {}".format(imgs.shape))

        # make data to save it to json
        data = [{} for _ in range(imgs.shape[0])]

        for i in range(imgs.shape[0]):
            data[i]["cctv_id"] = cctv_id
            data[i]["video_name"] = params.video_name
            data[i]["frame_id"] = i
            data[i]["width"] = original_size[1]
            data[i]["height"] = original_size[0]
            data[i]["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data[i]["annotation"] = []
            data[i]["banner_annotation"] = []

        # DET
        det_results = banDet(self.det_model, self.test_pipeline, imgs_array,
                             imgs, params, data)

        cropped_imgs, indices, four_corners = crop_n_resize(imgs, det_results)

        if params.verbose == True:
            logging.info("DET time : {}".format(
                time.time() - preprocessing_time))
            det_time = time.time()

        # Extract Cetner
        pred_points = extract_center(det_results)

        banner_num = np.zeros(imgs.shape[0], dtype=np.uint)
        banner_num_str = banner_num.copy()

        for point in pred_points:
            banner_num[point[0]] += 1

        if pred_points == []:
            for i in range(len(data)):
                with open(os.path.join(save_root_path, "{}.json".format(i)), "w", encoding='utf-8') as f:
                    f.write(json.dumps(
                        data[i], ensure_ascii=False, indent=4, cls=NumpyEncoder))
            return data

        # selected_images, selected_points = select_images_based_on_pred_points(
        #     imgs, pred_points)

        # selected_images, selected_points = selected_images.to(
        #     f"cuda:{self.gpu_ids['sam']}"), selected_points.to(f"cuda:{self.gpu_ids['sam']}")

        # print("Step 2: Banner Segmentation running...")
        # seg_results = []
        # for i in range(selected_images.shape[0]):
        #     selected_image = selected_images[i].unsqueeze(0)
        #     selected_point = selected_points[i].unsqueeze(0)

        #     seg_result, scores = banSeg(
        #         self.seg_model, selected_image, selected_point, pred_points[i][0], banner_num, params)
        #     seg_results.extend(seg_result)

        # if params.verbose == True:
        #     logging.info("SEG time : {}".format(time.time() - det_time))
        #     seg_time = time.time()

        # Warp
        # warp_out, inverse_transform_matrix, four_corners, warped_idxes = banWarp(
        #     seg_results, selected_images, pred_points, params)

        for i, (four_corner, warped_idx) in enumerate(zip(four_corners, indices)):
            banner_annotation = {}
            banner_annotation["points"] = four_corner
            banner_annotation["center"] = [np.array(
                four_corner).mean(axis=0).tolist()]
            banner_annotation["banner_id"] = len(
                data[warped_idx]["banner_annotation"])
            banner_annotation["num_instances"] = 0
            data[warped_idx]["banner_annotation"].append(banner_annotation)

        # if params.verbose == True:
        #     logging.info("Warp time : {}".format(time.time() - seg_time))
        #     warp_time = time.time()

        # STD
        std_results, poly_results = banSTD(
            self.std_model, cropped_imgs, params, gpu_id=f"cuda:{self.gpu_ids['std']}")

        # if params.verbose == True:
        #     logging.info("STD time : {}".format(time.time() - warp_time))
        #     std_time = time.time()

        # STR
        print("Step 3: Scene Text Recognition running...")
        str_results = []
        for k, (std_result, poly_result) in enumerate(zip(std_results, poly_results)):
            if std_result == []:
                banner_num_str[indices[k]] += 1
                continue
            for i in range(len(std_result)):
                text_image = torch.Tensor(
                    std_result[i]).to(f"cuda:{self.gpu_ids['str']}") / 255.0
                text_image = self.resize(text_image.permute(2, 0, 1))

                text_image = self.str_preprocess(text_image)

                std_result[i] = text_image

            std_result = torch.stack(std_result).to(f"cuda:{self.gpu_ids['str']}")

            res, probs = self.str_model.test_step((std_result, "test"), -1)

            str_result = []
            for i, (text, poly, prob) in enumerate(zip(res, poly_result, probs)):
                x1, y1, x2, y2 = poly

                x1 = x1 + four_corners[k][0][0]
                y1 = y1 + four_corners[k][0][1]
                x2 = x2 + four_corners[k][0][0]
                y2 = y2 + four_corners[k][0][1]

                # points = np.array([[[min_width, min_height]],
                #                    [[min_width, max_height]],
                #                    [[max_width, min_height]],
                #                    [[max_width, max_height]]], dtype=np.float32)

                # 각 점에 대해 perspectiveTransform을 호출합니다.
                # transformed_points = cv2.perspectiveTransform(
                #     points, inverse_transform_matrix[k])

                # 변환된 점들을 사용하여 xmin, ymin, xmax, ymax를 계산합니다.
                # xmin = np.min(points[:, :, 0])
                # ymin = np.min(points[:, :, 1])
                # xmax = np.max(points[:, :, 0])
                # ymax = np.max(points[:, :, 1])

                str_result.append([text, 
                                    [(x1 + x2) / 2, 
                                     (y1 + y2) / 2], 
                                    banner_num_str[indices[k]].item(),
                                    prob.prod().item()])

                annotation = {}
                annotation["text"] = text
                annotation["points"] = [[x1, y2], [x2, y1]]
                annotation["instance_id"] = data[indices[k]
                                                 ]["banner_annotation"][banner_num_str[indices[k]]]["num_instances"]
                annotation["banner_id"] = banner_num_str[indices[k]].item()
                data[indices[k]]["banner_annotation"][banner_num_str[indices[k]]
                                                           ]["num_instances"] += 1
                annotation["score_cls"] = prob.prod().item()
                data[indices[k]]["annotation"].append(annotation)

            banner_num_str[indices[k]] += 1
            str_results.append(str_result)

        if params.scene_text_detection.save_results or params.scene_text_recognition.save_results:
            for i in range(len(data)):
                img = imgs[i].detach().permute(1, 2, 0).cpu().numpy() * 255
                img = img.astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.resize(img, (data[i]['width'], data[i]['height']))

                if params.scene_text_detection.save_results:
                    img_std = img.copy()
                    for annotation in data[i]["annotation"]:
                        # remake poly to points
                        points = annotation["points"]
                        bb.add(img_std, points[0][0]/1024*data[i]['width'], points[1][1]/1024*data[i]['height'],
                               points[1][0]/1024*data[i]['width'], points[0][1]/1024*data[i]['height'])

                    filename = params.out_img_filename.split(".")
                    filename[0] += f"_std{i}"
                    filename = ".".join(filename)
                    img_std = cv2.cvtColor(img_std.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(save_root_path, filename), img_std)

                if params.scene_text_recognition.save_results:
                    for annotation in data[i]["annotation"]:
                        # remake poly to points
                        points = annotation["points"]
                        bb.add(img, points[0][0]/1024*data[i]['width'], points[1][1]/1024*data[i]['height'],
                               points[1][0]/1024*data[i]['width'], points[0][1]/1024*data[i]['height'], label=annotation["text"])

                    # filename = params.out_img_filename.split(".")
                    # filename[0] += f"_str{i}"
                    # filename = ".".join(filename)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    filename = params.out_img_filename.split(".")
                    filename[0] += f"_str{i}"
                    filename = ".".join(filename)
                    cv2.imwrite(os.path.join(save_root_path, filename), img)

        # if params.verbose == True:
        #     logging.info("STR time : {}".format(time.time() - std_time))
        
        # Ordering text preserve contextual knowledge
        print("Step 4: Ordering Text running...")
        contours = banContour(self.ctr_model, imgs, params)
        # print(str_results)
        # print("Data type of the str_results: ", type(str_results))
        ordered_text = union_text_rev(str_results, contours)
        # print(ordered_text)
        # ordered_text = str_results 
        # print("Data type of the ordered_text: ", type(ordered_text))

        # Refine and Classification
        if ordered_text != [{}]: 
            print("Step 5: Refine and Classification running...")
            refine_cls = banRefAndCls(self.ref_cls_model, f"cuda:{self.gpu_ids['ref_cls']}", ordered_text)
            
        color_dict = {
            "공공": "green",
            "민간": "red",
            "정당": "aqua",
            "알수없음": "yellow"
        }

        # save data to json
        for i in range(len(data)):
            # Resize points to original size of image.
            for annotation in data[i]["annotation"]:
                annotation["points"] = [[int(point[0] * data[i]["width"] / 1024), int(
                    point[1] * data[i]["height"] / 1024)] for point in annotation["points"]]
            for banner_annotation in data[i]["banner_annotation"]:
                banner_annotation["points"] = [[int(point[0] * data[i]["width"] / 1024), int(
                    point[1] * data[i]["height"] / 1024)] for point in banner_annotation["points"]]
                banner_annotation["center"] = [[int(point[0] * data[i]["width"] / 1024), int(
                    point[1] * data[i]["height"] / 1024)] for point in banner_annotation["center"]]
            
            if ordered_text != [{}]:
                for item in refine_cls[i]:
                    b_id = item["banner_id"]
                    ref_txt = item["refined_text"]
                    ban_cls = item["class"]

                    ordered_text[i][b_id]["ref_txt"] = ref_txt
                    ordered_text[i][b_id]["ban_cls"] = ban_cls

                data[i]["summary"] = ordered_text[i]

            filename = params.out_metadata.split(".")
            filename[0] += f"{i}"
            filename = ".".join(filename)
            with open(os.path.join(save_root_path, filename), "w", encoding='utf-8') as f:
                f.write(json.dumps(
                    data[i], ensure_ascii=False, indent=4, cls=NumpyEncoder))
                
            img = cv2.imread(data[i]['video_name'])
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
            for xyxy in data[i]['banner_annotation']:
                    _, x1, y1, _ = xyxy['points']
                    try:
                        bb.add(img, x1[0], x1[1], y1[0], y1[1], format(f"{data[i]['summary'][xyxy['banner_id']]['ban_cls']}"), color=color_dict[data[i]['summary'][xyxy['banner_id']]['ban_cls']])
                    except:
                        pass
            filename = params.out_img_filename.split(".")
            filename[0] += f"_cls{i}"
            filename = ".".join(filename)
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(
                params.save_root_path, filename), img)
            
        if params.verbose == True:
            logging.info("Inference time : {}".format(
                time.time() - start_time))
            logging.info("End time : {}".format(datetime.now()))
            logging.info(
                "=====================================================================")
        
        return data

    def unload_model(self):
        del self.det_model
        # del self.sam_model
        del self.std_model
        del self.str_model
        del self.ctr_model
        del self.ref_cls_model_model

        gc.collect()

        if "cuda" in self.device:
            torch.cuda.empty_cache()


class config:
    params = {
        "save_root_path": "./output/",
        "font_path": "malgun.ttf",
        "sample_rate": 1,
        "camid": 0,
        "verbose": False,
        "preprocess": {
            "save_results": False
        },
        "detection": {
            "conf_thres": 0.3,
            "iou_thres": 0.6,
            "save_results": True
        },
        "segmentation": {
            "mask_threshold": 3.0,
            "save_results": False
        },
        "warping": {
            "save_results": False
        },
        "scene_text_detection": {
            "text_threshold": 0.7,
            "link_threshold": 0.4,
            "low_text": 0.4,
            "poly": False,
            "canvas_size": 1280,
            "mag_ratio": 1.5,
            "save_results": True
        },
        "scene_text_recognition": {
            "save_results": True
        },
        "contour":{
            "save_results":False
        }
    }


def banner_init_gpu(gpu_id, config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        conf = json.load(file)

    det_model, test_pipeline = load_det_model(gpu_id, conf["det"]["model_path"])
    # sam_model = load_sam_model(gpu_id, conf["sam"]["model_path"])
    std_model = load_std_model(gpu_id, conf["std"]["model_path"])
    str_model = load_str_model(gpu_id, conf["str"]["model_path"])
    ctr_model = load_ctr_model(gpu_id, conf["ctr"]["model_path"])
    ref_cls_model, args = load_ref_cls_model(gpu_id, conf["ref_cls"]["model_path"])
    return args, (det_model, test_pipeline), std_model, str_model, ctr_model, ref_cls_model

def banner_analysis(cctv_id, in_img_filename, out_img_filename, out_metadata, models, gpu_id):
    params = EasyDict(config.params)
    params.video_name = in_img_filename
    params.out_img_filename = out_img_filename
    params.out_metadata = out_metadata
    
    params.save_path = in_img_filename.split(".")[0].split("/")[-1] # datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")[:-3]
    bb._FONT = ImageFont.truetype(params.font_path, 15)

    Ban = Banner(gpu_id)
    Ban.inference(cctv_id, params.video_name, models, params)

def banner_release_gpu(gpu_id, models):
    for model in models: del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    cctv_id = 2
    gpu_id = 7
    conf_path = "./tmpv5.json"
    args, *ban_models = banner_init_gpu(gpu_id, conf_path)
    
    banner_analysis(cctv_id, args.img_name, "out.jpg", "out.json", ban_models, gpu_id)
    banner_release_gpu(gpu_id, ban_models)
