import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import Optional


@dataclass
class Configs:
  patchSize:int = 16
  embeddingChannels:int = 32
  batchSize: int = 2
  layerNormEps: int = 1e-5
  intermediateEmbedding :int = 32 * 4
  numLayers: int = 6
  numHeads: int = 4
  dropoutRate:float = 0.5
  training: bool = True
  imageDim:int = 28
  device:str = 'cuda'

class SiglipVisionEmbedding(nn.Module):

  def __init__(self, config: Configs, layerNormEps: int = 1e-5, attentionDropout: int = 0.1, numImageTokens: int = 0):
    super().__init__()

    self.patchConv = nn.Conv2d(3, config.embeddingChannels,  config.patchSize, config.patchSize) #assuming rgb image
    self.numPositions = torch.arange((config.imageDim // config.patchSize)** 2).to(device = config.device) #after flattening, this is how many tokens to attach positional embedding to
    self.positionEmbeddings = nn.Embedding(self.numPositions.shape[0], config.embeddingChannels) #just embedding the index of patch
  def forward(self, x):
    x = self.patchConv(x)
    x = torch.flatten(x, 2) #B, C, H * W
    x = x.transpose(1, 2) #B, H * W, C (because we want a sequence of embeddings listed in order)
    x = self.positionEmbeddings(self.numPositions) + x

    return x

class SiglipFeedForward(nn.Module):
  def __init__(self, config: Configs):
    super().__init__()
    self.fc1 = nn.Linear(config.embeddingChannels, config.intermediateEmbedding)
    self.fc2 = nn.Linear(config.intermediateEmbedding, config.embeddingChannels)

  def forward(self, x):
    x = self.fc1(x)
    x = self.fc2(F.gelu(x, approximate = 'tanh'))
    return x

class SiglipSelfAttention(nn.Module):
  def __init__(self, config: Configs):
    super().__init__()
    self.config = config
    self.training = config.training
    self.wq = nn.Linear(config.embeddingChannels, config.embeddingChannels)
    self.wk = nn.Linear(config.embeddingChannels, config.embeddingChannels)
    self.wv = nn.Linear(config.embeddingChannels, config.embeddingChannels)
    self.wo = nn.Linear(config.embeddingChannels, config.embeddingChannels)

  def forward(self, x):

    config = self.config
    batchSize,seqLength, numFeatures = x.shape

    xq = self.wq(x)
    xk = self.wk(x)
    xv = self.wv(x)

    # B, seqLength, embeddingDim > B, numHeads, seqLength, dimPerHead
    xq = xq.view(batchSize, seqLength, config.numHeads, numFeatures // config.numHeads).transpose(1,2)
    xk = xk.view(batchSize, seqLength, config.numHeads, numFeatures // config.numHeads).transpose(1,2)
    xv = xv.view(batchSize, seqLength, config.numHeads, numFeatures // config.numHeads).transpose(1,2)

    attWeights = torch.matmul(xq, xk.transpose(2,3)) / torch.sqrt(torch.tensor(numFeatures // config.numHeads)) # xq matmul xk.T > B, numHeads, seqLength, seqLength
    attWeights = F.dropout(attWeights, p = self.config.dropoutRate, training = self.training)
    attWeights = F.softmax(attWeights, dim = -1)

    o = torch.matmul(attWeights, xv).transpose(1,2) # B, seqLength, numHeads, dimPerHead
    # using reshape instead of view because of some memory issue that might occur
    o = o.reshape(batchSize, seqLength, numFeatures)# B, seqLength, totalDims

    return self.wo(o.contiguous())

class SiglipVisionEncoder(nn.Module):
  def __init__(self, config: Configs):
    super().__init__()
    self.embedDim = config.embeddingChannels
    self.norm1 = nn.LayerNorm(self.embedDim, eps = config.layerNormEps)
    self.att = SiglipSelfAttention(config)
    self.norm2 = nn.LayerNorm(self.embedDim, eps = config.layerNormEps)
    self.FF = SiglipFeedForward(config)


  def forward(self, x):
    x = self.norm1(x)
    x = self.att(x)
    x = x + self.norm2(x)
    x = self.FF(x)
    return x

class SiglipTransformer(nn.Module):
  def __init__(self, config: Configs):
    super().__init__()
    self.embed = SiglipVisionEmbedding(config) 
    self.layers = nn.ModuleList([SiglipVisionEncoder(config) for l in range(config.numLayers)])
    self.normFinal = nn.LayerNorm(config.embeddingChannels, eps = config.layerNormEps)

  def forward(self, x):
    x = self.embed(x)

    for l in self.layers:
      x = l(x)

    x = self.normFinal(x)

    return x
    
    
@dataclass
class TextConfigs:
  vocabSize:int = 0
  visionEmbeddingChannels:int = 0
  textEmbeddingChannels:int = 0
  batchSize: int = 2
  rmsNormEps: int = 1e-5
  intermediateEmbedding :int = 32 * 4
  numLayers: int = 6
  numHeads: int = 4
  dropoutRate:float = 0.5
  training:bool = True
  base: int = 1000
  batchSize:int  = 8
  maxSeqLength:int = 32
  dropoutRate:float = 0.5
  device:str = 'cpu'

class MultiModalProjector(nn.Module):
  def __init__(self, config: TextConfigs):
    super().__init__()
    self.project = nn.Linear(config.visionEmbeddingChannels, config.textEmbeddingChannels)

  def forward(self, x):
    return self.project(x)
    
class RotaryEmbed(nn.Module):
  def __init__(self, config: TextConfigs):
    super().__init__()
    base = 10000
    self.freqs = 1.0 / (torch.pow(base, torch.arange(0, config.textEmbeddingChannels // config.numHeads, 2) / 2)).to(device = config.device)

  #======

  def rotate_half(self, x):
      x1 = x[..., : x.shape[-1] // 2] # Takes the first half of the last dimension
      x2 = x[..., x.shape[-1] // 2 :] # Takes the second half of the last dimension
      #not exactly the way it is in the paper as this simply gives something like the [-x_n, -x_n-1, ..., x_n/2 - 1, x_n/2], instead of the formula in the paper
      return torch.cat((-x2, x1), dim=-1)


  def apply_rotary_pos_emb(self, q, k, cos, sin, unsqueeze_dim=1):
      cos = cos.unsqueeze(unsqueeze_dim) # Add the head dimension
      sin = sin.unsqueeze(unsqueeze_dim) # Add the head dimension
      # Apply the formula (34) of the Rotary Positional Encoding paper.
      q_embed = (q * cos) + (self.rotate_half(q) * sin) #[[cos, -sin], [sin, cos]]
      k_embed = (k * cos) + (self.rotate_half(k) * sin)
      return q_embed, k_embed

  #=======

  def forward(self, positionIds, q, k):
    freqs = self.freqs[None, :, None].expand(positionIds.shape[0], -1, 1).float() 
    positionIds = positionIds[:,None,:].float()
    emb = (freqs @ positionIds).transpose(1, 2) #B, SeqLen, headDim//2
    emb = torch.cat((emb, emb), dim = -1)
    qPosEmb, kPosEmb = self.apply_rotary_pos_emb(q, k, emb.cos(), emb.sin(), unsqueeze_dim=1)
    return qPosEmb, kPosEmb

class GemmaFeedForward(nn.Module):
    def __init__(self, config: TextConfigs):
        super().__init__()
        self.config = config
        self.inputSize = config.textEmbeddingChannels
        self.intermediateSize = config.intermediateEmbedding
        self.gate_proj = nn.Linear(self.inputSize, self.intermediateSize, bias=False)
        self.up_proj = nn.Linear(self.inputSize, self.intermediateSize, bias=False)
        self.down_proj = nn.Linear(self.intermediateSize, self.inputSize, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))

class SelfAttentionText(nn.Module):
  def __init__(self, config: TextConfigs):
    super().__init__()
    self.config = config
    self.wq = nn.Linear(config.textEmbeddingChannels, config.textEmbeddingChannels)
    self.wk = nn.Linear(config.textEmbeddingChannels, config.textEmbeddingChannels)
    self.wv = nn.Linear(config.textEmbeddingChannels, config.textEmbeddingChannels)
    self.wo = nn.Linear(config.textEmbeddingChannels, config.textEmbeddingChannels)
    self.rotaryEmb = RotaryEmbed(config)

  def forward(self, x, position_ids, currPositionInSeq = 0, attMask=None):

    batchSize,seqLength, features = x.shape

    assert features % self.config.numHeads == 0

    q = self.wq(x).view(batchSize, seqLength, self.config.numHeads, features // self.config.numHeads).transpose(1, 2) #b, h, l, hd
    k = self.wk(x).view(batchSize, seqLength, self.config.numHeads, features // self.config.numHeads).transpose(1, 2) #b, h, l, hd
    v = self.wv(x).view(batchSize, seqLength, self.config.numHeads, features // self.config.numHeads).transpose(1, 2) #b, h, l, hd
    q, k = self.rotaryEmb(position_ids , q, k)
    attMatrix = torch.matmul(q, k.transpose(2, 3)) #b, h, l, l
    #dont really need mask for this application because there is no way to access future tokens in a given multimodal input
    #attMatrix = attMatrix + attMask
    
    out = torch.softmax(attMatrix, dim = -1)
    attMatrix = F.dropout(attMatrix, p = self.config.dropoutRate, training = self.config.training)
    out = torch.matmul(out, v).transpose(1,2)
    
    out = out.reshape(batchSize, seqLength, features) #had to change it from out.view because it might cause memory issues due to not being contiguous
    out = self.wo(out)
    
    return out.squeeze(1) 

class TextDecoder(nn.Module):
  def __init__(self, config: TextConfigs):
    super().__init__()
    self.att = SelfAttentionText(config)
    self.norm1 = nn.RMSNorm( config.textEmbeddingChannels, eps = config.rmsNormEps) #batchSize, seqLength = 1, features
    self.FF = GemmaFeedForward(config)
    self.norm2 = nn.RMSNorm(config.textEmbeddingChannels, eps = config.rmsNormEps)

  def forward(self, x, currPos):
    x = self.att(x,currPos) + self.norm1(x)
    x = self.FF(x) + self.norm2(x)
    return x

class MultiModalTransformer(nn.Module):
  def __init__(self, config: TextConfigs):
    super().__init__()
    self.decoderLayers = nn.ModuleList([TextDecoder(config) for layer_idx in range(config.numLayers)])
    self.normFinal = nn.RMSNorm( config.textEmbeddingChannels, eps = config.rmsNormEps)
    self.logits = nn.Linear(config.textEmbeddingChannels, config.vocabSize)

  def forward(self, attention_mask,position_ids,multiModalToLogits): #miniature kv cache is implemented in the function itself
    for nextLayer in self.decoderLayers:
        multiModalToLogits = nextLayer(multiModalToLogits, position_ids)

    multiModalToLogits = self.normFinal(multiModalToLogits)
    multiModalToLogits = self.logits(multiModalToLogits)
    
    return multiModalToLogits
    
class MultiModalPipeline(nn.Module):
  def __init__(self, textConfigs: TextConfigs, visionConfigs: Configs, tokenMap):
    super().__init__()
    self.textConfigs = textConfigs
    self.visionConfigs = visionConfigs
    self.embed = nn.Embedding(textConfigs.vocabSize + 1, textConfigs.textEmbeddingChannels).to(device=textConfigs.device)
    self.imageComponent = SiglipTransformer(visionConfigs).to(device=textConfigs.device)
    self.projected = MultiModalProjector(textConfigs).to(device=textConfigs.device)
    self.logits = MultiModalTransformer(textConfigs).to(device=textConfigs.device)
    self.tokenMap = tokenMap

  def _merge_input_ids_with_image_features(self, image_features: torch.Tensor, inputs_embeds: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, image_token_index: int, pad_token_id: int, hidden_size: float):
      _, _, embed_dim = image_features.shape
      batch_size, sequence_length = input_ids.shape
      dtype, device = inputs_embeds.dtype, inputs_embeds.device
      # Shape: [Batch_Size, Seq_Len, Hidden_Size]
      scaled_image_features = image_features / (hidden_size**0.5)

      # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens.
      final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
      final_embedding[:, scaled_image_features.shape[1]:] = inputs_embeds[:, scaled_image_features.shape[1]:]
      final_embedding[:, :scaled_image_features.shape[1]] = scaled_image_features


      #### CREATE THE ATTENTION MASK ####

      min_dtype = torch.finfo(dtype).min
      q_len = inputs_embeds.shape[1]

      # Do not mask any token, because we're in the prefill phase
      # This only works when we have no padding
      causal_mask = torch.full(
          (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
      )

      # Add the head dimension
      causal_mask = causal_mask.unsqueeze(1)
      position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

      return final_embedding, causal_mask, position_ids

  def forward(self, input_ids: torch.LongTensor = None,pixel_values: torch.FloatTensor = None,attention_mask: Optional[torch.Tensor] = None,kv_cache = None, labels = None):
    inputs_embeds = self.embed(input_ids)
    h = self.imageComponent(pixel_values)
    h = self.projected(h)
    inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(h, inputs_embeds, input_ids, attention_mask, self.tokenMap['<image>'], 500, self.visionConfigs.intermediateEmbedding)

    finalLogits = self.logits(attention_mask=attention_mask,position_ids=position_ids,multiModalToLogits=inputs_embeds)
    finalLogits = finalLogits[:, -1]
    labels = labels.reshape(labels.shape[0] * labels.shape[1])
    pred = finalLogits

    loss = F.cross_entropy(pred.to(device= self.textConfigs.device), labels.to(device= self.textConfigs.device))

    return pred, loss
