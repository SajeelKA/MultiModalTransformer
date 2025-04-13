# MultiModalTransformer

## Instructions

#### This project was trained on Google Colaboratory, so a path needs to be made under '/content/drive/MyDrive/MultiModal' in your Google Drive. This path will act as the root directory

#### Next, you can open the runTrainingScript.ipynb file and login to huggingface through the command-line using a token generated from a huggingface account

#### After that, you can run the cell which runs the file "Train.py" to be able to run the training. 

### Hyper-parameters that can be input to the train script:
  
  #### "--batchSize", required = False, type=int, default = 1  
  #### "--learningRate", required = False, type=float, default = 1e-5
  #### "--epochs", required = False, type=int, default = 5
  #### "--parallelHeads", required = False, type=int, default = 4
  #### "--nImageLayers", required = False, type=int, default = 1 
  #### "--patchSize", required = False, type=int, default = 1
  #### "--nTextLayers", required = False, type=int, default = 2
  #### "--imageChannels", required = False, type=int, default = 4  
  #### "--textChannels", required = False, type=int, default = 512
  #### "--usedProcessor", required = False, default = 'cuda'
  #### "--savePath", required = False, default = None

# Parameters Per Component and Sample Flop Calculation:

### Given:
### Batch Size = 2
### Image embedding channels = 512
### Image patch size = 1
### Image component number of layers = 2
### Text vocabulary size = 257153
### Text embedding channels = 128
### Text number of layers = 1

### and "T" represents sequence length

257153 x 128 = 32915584 total parameters for embed.weight
0 flops

512 x 3 x 1 x 1 = 1536 total parameters for imageComponent.embed.patchConv.weight
1536 x 2 flops = 6144 x T (3 in channels, 512 out channels, 1 x 1 kernel)

512 total parameters for imageComponent.embed.patchConv.bias
1024 x T  flops

784 x 512 = 401408 total parameters for imageComponent.embed.positionEmbeddings.weight
820,816 x T flops (adding positional value to image tokens)

512 total parameters for imageComponent.layers.0.norm1.weight
(4 x 512 flops) x 2 x T = 4096 x T (512 dims squaring (512 flops), summation (512 flops), division of input by rms (512 flops) multiply by learnable gamma parameter (512 flops))

512 total parameters for imageComponent.layers.0.norm1.bias
1024 x T flops

512 x 512 = 262144 total parameters for imageComponent.layers.0.att.wq.weight
512 x 512 x 2 flops x 2 x T = 1,048,576 x T flops

512 total parameters for imageComponent.layers.0.att.wq.bias
1024 x T flops

512 x 512 = 262144 total parameters for imageComponent.layers.0.att.wk.weight
512 x 512 x 2 flops x 2 x T = 1,048,576 x T

512 total parameters for imageComponent.layers.0.att.wk.bias
1024 x T flops

512 x 512 = 262144 total parameters for imageComponent.layers.0.att.wv.weight
512 x 512 x 2 flops x 2 x T = 1,048,576 x T

512 total parameters for imageComponent.layers.0.att.wv.bias
1024 x T

512 x 512 = 262144 total parameters for imageComponent.layers.0.att.wo.weight
512 x 512 x 2 flops x 2 = 1,048,576 x T

512 total parameters for imageComponent.layers.0.att.wo.bias
1024 flops x T

512 total parameters for imageComponent.layers.0.norm2.weight
(4 x 512 flops) x 2 x T = 4096 x T (512 dims squaring (512 flops), summation (512 flops), division of input by rms (512 flops) multiply by learnable gamma parameter (512 flops))

512 total parameters for imageComponent.layers.0.norm2.bias
1024 flops x T

128 x 512 = 65536 total parameters for imageComponent.layers.0.FF.fc1.weight
128 x 512 x 2 x 2 x T = 262,144 x T flops

128 total parameters for imageComponent.layers.0.FF.fc1.bias
256 x T flops

512 x 128 = 65536 total parameters for imageComponent.layers.0.FF.fc2.weight
512 x 128 x 2 x 2 x T = 262,144 x T flops

512 total parameters for imageComponent.layers.0.FF.fc2.bias
1024 x T flops

512 total parameters for imageComponent.layers.1.norm1.weight
(4 x 512 flops) x 2 x T = 4096 x T (512 dims squaring (512 flops), summation (512 flops), division of input by rms (512 flops) multiply by learnable gamma parameter (512 flops))

512 total parameters for imageComponent.layers.1.norm1.bias
1024 x T flops

512 x 512 = 262144 total parameters for imageComponent.layers.1.att.wq.weight
512 x 512 x 2 flops x 2 x T = 1,048,576 x T

512 total parameters for imageComponent.layers.1.att.wq.bias
1024 x T flops

512 x 512 = 262144 total parameters for imageComponent.layers.1.att.wk.weight
512 x 512 x 2 flops x 2 x T = 1,048,576 x T

512 total parameters for imageComponent.layers.1.att.wk.bias
1024 x T flops

512 x 512 = 262144 total parameters for imageComponent.layers.1.att.wv.weight
512 x 512 x 2 flops x 2 x T = 1,048,576 x T

512 total parameters for imageComponent.layers.1.att.wv.bias
1024 x T flops

512 x 512 = 262144 total parameters for imageComponent.layers.1.att.wo.weight
512 x 512 x 2 flops x 2 x T = 1,048,576 x T

512 total parameters for imageComponent.layers.1.att.wo.bias
1024 x T flops

512 total parameters for imageComponent.layers.1.norm2.weight
(4 x 512 flops) x 2 x T = 4096 x T (512 dims squaring (512 flops), summation (512 flops), division of input by rms (512 flops) multiply by learnable gamma parameter (512 flops))

512 total parameters for imageComponent.layers.1.norm2.bias
1024 x T flops

128 x 512 = 65536 total parameters for imageComponent.layers.1.FF.fc1.weight
128 x 512 x 2 x 2 x T = 262,144 x T flops

128 total parameters for imageComponent.layers.1.FF.fc1.bias
256 x T flops

512 x 128 = 65536 total parameters for imageComponent.layers.1.FF.fc2.weight
512 x 128 x 2 x 2 x T = 262,144 x T flops

512 total parameters for imageComponent.layers.1.FF.fc2.bias
1024 x T flops

512 total parameters for imageComponent.normFinal.weight
(4 x 512 flops) x 2 x T = 4096 x T (512 dims squaring (512 flops), summation (512 flops), division of input by rms (512 flops) multiply by learnable gamma parameter (512 flops))

512 total parameters for imageComponent.normFinal.bias
1024 x T flops

128 x 512 = 65536 total parameters for projected.project.weight
128 x 512 x 2 x 2 x T = 262,144 x T flops

128 total parameters for projected.project.bias
256 x T flops

128 x 128 = 16384 total parameters for logits.decoderLayers.0.att.wq.weight
128 x 128 x 2 x 2 x T = 65,536 x T flops

128 total parameters for logits.decoderLayers.0.att.wq.bias
256 x T flops

128 x 128 = 16384 total parameters for logits.decoderLayers.0.att.wk.weight
128 x 128 x 2 x 2 x T = 65,536 x T flops

128 total parameters for logits.decoderLayers.0.att.wk.bias
256 x T flops

128 x 128 = 16384 total parameters for logits.decoderLayers.0.att.wv.weight
128 x 128 x 2 x 2 x T = 65,536 x T flops

128 total parameters for logits.decoderLayers.0.att.wv.bias
256 x T flops

128 x 128 = 16384 total parameters for logits.decoderLayers.0.att.wo.weight
128 x 128 x 2 x 2 x T = 65,536 x T flops

128 total parameters for logits.decoderLayers.0.att.wo.bias
256 x T flops

128 total parameters for logits.decoderLayers.0.norm1.weight
(4 x 128 flops) x 2 x T = 1024 x T (128 dims squaring (128 flops), summation (128 flops), division of input by rms (128 flops) multiply by learnable gamma parameter (128 flops))

128 x 128 = 16384 total parameters for logits.decoderLayers.0.FF.gate_proj.weight
128 x 128 x 2 x 2 x T = 65,536 x T flops

128 x 128 = 16384 total parameters for logits.decoderLayers.0.FF.up_proj.weight
128 x 128 x 2 x 2 x T = 65,536 x T flops

128 x 128 = 16384 total parameters for logits.decoderLayers.0.FF.down_proj.weight
128 x 128 x 2 x 2 x T = 65,536 x T flops

128 total parameters for logits.decoderLayers.0.norm2.weight
(4 x 128 flops) x 2 x T = 1024 x T (128 dims squaring (128 flops), summation (128 flops), division of input by rms (128 flops) multiply by learnable gamma parameter (128 flops))

128 total parameters for logits.normFinal.weight
(4 x 128 flops) x 2 x T = 1024 x T (128 dims squaring (128 flops), summation (128 flops), division of input by rms (128 flops) multiply by learnable gamma parameter (128 flops))

257153 x 128 = 32915584 total parameters for logits.logits.weight
257153 x 128 x 2 x 2 x T = 131,662,336 flops

257153 total parameters for logits.logits.bias
514,306 x T flops

================== total flops =============== 

140,841,810 x T




