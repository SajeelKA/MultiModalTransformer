Sample of model Architecture:
================================

MultiModalPipeline(
  (embed): Embedding(257153, 128) 
  (imageComponent): SiglipTransformer(
    (embed): SiglipVisionEmbedding(
      (patchConv): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
      (positionEmbeddings): Embedding(784, 128)
    )
    (layers): ModuleList(
      (0): SiglipVisionEncoder(
        (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (att): SiglipSelfAttention(
          (wq): Linear(in_features=128, out_features=128, bias=True)
          (wk): Linear(in_features=128, out_features=128, bias=True)
          (wv): Linear(in_features=128, out_features=128, bias=True)
          (wo): Linear(in_features=128, out_features=128, bias=True)
        )
        (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (FF): SiglipFeedForward(
          (fc1): Linear(in_features=128, out_features=128, bias=True)
          (fc2): Linear(in_features=128, out_features=128, bias=True)
        )
      )
    )
    (normFinal): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
  )
  (projected): MultiModalProjector(
    (project): Linear(in_features=128, out_features=128, bias=True)
  )
  (logits): MultiModalTransformer(
    (decoderLayers): ModuleList(
      (0): TextDecoder(
        (att): SelfAttentionText(
          (wq): Linear(in_features=128, out_features=128, bias=True)
          (wk): Linear(in_features=128, out_features=128, bias=True)
          (wv): Linear(in_features=128, out_features=128, bias=True)
          (wo): Linear(in_features=128, out_features=128, bias=True)
          (rotaryEmb): RotaryEmbed()
        )
        (norm1): RMSNorm((128,), eps=1e-05, elementwise_affine=True)
        (FF): GemmaFeedForward(
          (gate_proj): Linear(in_features=128, out_features=128, bias=False)
          (up_proj): Linear(in_features=128, out_features=128, bias=False)
          (down_proj): Linear(in_features=128, out_features=128, bias=False)
        )
        (norm2): RMSNorm((128,), eps=1e-05, elementwise_affine=True)
      )
    )
    (normFinal): RMSNorm((128,), eps=1e-05, elementwise_affine=True)
    (logits): Linear(in_features=128, out_features=257153, bias=True)
  )
)

=========================================
Model Parameters for above architecture:
=========================================

embed.weight torch.Size([257153, 128])
257153 x 128 = 32915584 total parameters for embed.weight

imageComponent.embed.patchConv.weight torch.Size([512, 3, 1, 1])
512 x 3 x 1 x 1 = 1536 total parameters for imageComponent.embed.patchConv.weight

imageComponent.embed.patchConv.bias torch.Size([512])
512 total parameters for imageComponent.embed.patchConv.bias

imageComponent.embed.positionEmbeddings.weight torch.Size([784, 512])
784 x 512 = 401408 total parameters for imageComponent.embed.positionEmbeddings.weight

imageComponent.layers.0.norm1.weight torch.Size([512])
512 total parameters for imageComponent.layers.0.norm1.weight

imageComponent.layers.0.norm1.bias torch.Size([512])
512 total parameters for imageComponent.layers.0.norm1.bias

imageComponent.layers.0.att.wq.weight torch.Size([512, 512])
512 x 512 = 262144 total parameters for imageComponent.layers.0.att.wq.weight

imageComponent.layers.0.att.wq.bias torch.Size([512])
512 total parameters for imageComponent.layers.0.att.wq.bias

imageComponent.layers.0.att.wk.weight torch.Size([512, 512])
512 x 512 = 262144 total parameters for imageComponent.layers.0.att.wk.weight

imageComponent.layers.0.att.wk.bias torch.Size([512])
512 total parameters for imageComponent.layers.0.att.wk.bias

imageComponent.layers.0.att.wv.weight torch.Size([512, 512])
512 x 512 = 262144 total parameters for imageComponent.layers.0.att.wv.weight

imageComponent.layers.0.att.wv.bias torch.Size([512])
512 total parameters for imageComponent.layers.0.att.wv.bias

imageComponent.layers.0.att.wo.weight torch.Size([512, 512])
512 x 512 = 262144 total parameters for imageComponent.layers.0.att.wo.weight

imageComponent.layers.0.att.wo.bias torch.Size([512])
512 total parameters for imageComponent.layers.0.att.wo.bias

imageComponent.layers.0.norm2.weight torch.Size([512])
512 total parameters for imageComponent.layers.0.norm2.weight

imageComponent.layers.0.norm2.bias torch.Size([512])
512 total parameters for imageComponent.layers.0.norm2.bias

imageComponent.layers.0.FF.fc1.weight torch.Size([128, 512])
128 x 512 = 65536 total parameters for imageComponent.layers.0.FF.fc1.weight

imageComponent.layers.0.FF.fc1.bias torch.Size([128])
128 total parameters for imageComponent.layers.0.FF.fc1.bias

imageComponent.layers.0.FF.fc2.weight torch.Size([512, 128])
512 x 128 = 65536 total parameters for imageComponent.layers.0.FF.fc2.weight

imageComponent.layers.0.FF.fc2.bias torch.Size([512])
512 total parameters for imageComponent.layers.0.FF.fc2.bias

imageComponent.layers.1.norm1.weight torch.Size([512])
512 total parameters for imageComponent.layers.1.norm1.weight

imageComponent.layers.1.norm1.bias torch.Size([512])
512 total parameters for imageComponent.layers.1.norm1.bias

imageComponent.layers.1.att.wq.weight torch.Size([512, 512])
512 x 512 = 262144 total parameters for imageComponent.layers.1.att.wq.weight

imageComponent.layers.1.att.wq.bias torch.Size([512])
512 total parameters for imageComponent.layers.1.att.wq.bias

imageComponent.layers.1.att.wk.weight torch.Size([512, 512])
512 x 512 = 262144 total parameters for imageComponent.layers.1.att.wk.weight

imageComponent.layers.1.att.wk.bias torch.Size([512])
512 total parameters for imageComponent.layers.1.att.wk.bias

imageComponent.layers.1.att.wv.weight torch.Size([512, 512])
512 x 512 = 262144 total parameters for imageComponent.layers.1.att.wv.weight

imageComponent.layers.1.att.wv.bias torch.Size([512])
512 total parameters for imageComponent.layers.1.att.wv.bias

imageComponent.layers.1.att.wo.weight torch.Size([512, 512])
512 x 512 = 262144 total parameters for imageComponent.layers.1.att.wo.weight

imageComponent.layers.1.att.wo.bias torch.Size([512])
512 total parameters for imageComponent.layers.1.att.wo.bias

imageComponent.layers.1.norm2.weight torch.Size([512])
512 total parameters for imageComponent.layers.1.norm2.weight

imageComponent.layers.1.norm2.bias torch.Size([512])
512 total parameters for imageComponent.layers.1.norm2.bias

imageComponent.layers.1.FF.fc1.weight torch.Size([128, 512])
128 x 512 = 65536 total parameters for imageComponent.layers.1.FF.fc1.weight

imageComponent.layers.1.FF.fc1.bias torch.Size([128])
128 total parameters for imageComponent.layers.1.FF.fc1.bias

imageComponent.layers.1.FF.fc2.weight torch.Size([512, 128])
512 x 128 = 65536 total parameters for imageComponent.layers.1.FF.fc2.weight

imageComponent.layers.1.FF.fc2.bias torch.Size([512])
512 total parameters for imageComponent.layers.1.FF.fc2.bias

imageComponent.normFinal.weight torch.Size([512])
512 total parameters for imageComponent.normFinal.weight

imageComponent.normFinal.bias torch.Size([512])
512 total parameters for imageComponent.normFinal.bias

projected.project.weight torch.Size([128, 512])
128 x 512 = 65536 total parameters for projected.project.weight

projected.project.bias torch.Size([128])
128 total parameters for projected.project.bias

logits.decoderLayers.0.att.wq.weight torch.Size([128, 128])
128 x 128 = 16384 total parameters for logits.decoderLayers.0.att.wq.weight

logits.decoderLayers.0.att.wq.bias torch.Size([128])
128 total parameters for logits.decoderLayers.0.att.wq.bias

logits.decoderLayers.0.att.wk.weight torch.Size([128, 128])
128 x 128 = 16384 total parameters for logits.decoderLayers.0.att.wk.weight

logits.decoderLayers.0.att.wk.bias torch.Size([128])
128 total parameters for logits.decoderLayers.0.att.wk.bias

logits.decoderLayers.0.att.wv.weight torch.Size([128, 128])
128 x 128 = 16384 total parameters for logits.decoderLayers.0.att.wv.weight

logits.decoderLayers.0.att.wv.bias torch.Size([128])
128 total parameters for logits.decoderLayers.0.att.wv.bias

logits.decoderLayers.0.att.wo.weight torch.Size([128, 128])
128 x 128 = 16384 total parameters for logits.decoderLayers.0.att.wo.weight

logits.decoderLayers.0.att.wo.bias torch.Size([128])
128 total parameters for logits.decoderLayers.0.att.wo.bias

logits.decoderLayers.0.norm1.weight torch.Size([128])
128 total parameters for logits.decoderLayers.0.norm1.weight

logits.decoderLayers.0.FF.gate_proj.weight torch.Size([128, 128])
128 x 128 = 16384 total parameters for logits.decoderLayers.0.FF.gate_proj.weight

logits.decoderLayers.0.FF.up_proj.weight torch.Size([128, 128])
128 x 128 = 16384 total parameters for logits.decoderLayers.0.FF.up_proj.weight

logits.decoderLayers.0.FF.down_proj.weight torch.Size([128, 128])
128 x 128 = 16384 total parameters for logits.decoderLayers.0.FF.down_proj.weight

logits.decoderLayers.0.norm2.weight torch.Size([128])
128 total parameters for logits.decoderLayers.0.norm2.weight

logits.normFinal.weight torch.Size([128])
128 total parameters for logits.normFinal.weight

logits.logits.weight torch.Size([257153, 128])
257153 x 128 = 32915584 total parameters for logits.logits.weight

logits.logits.bias torch.Size([257153])
257153 total parameters for logits.logits.bias

================== Total parameters =============== 69567891


