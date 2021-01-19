# YOLACT (You Only Look At CoefficienTs)

```{bash}
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─ResNet: 1-1                            [-1, 256, 138, 138]       --
|    └─Conv2d: 2-1                       [-1, 64, 275, 275]        9,408
|    └─BatchNorm2d: 2-2                  [-1, 64, 275, 275]        128
|    └─ReLU: 2-3                         [-1, 64, 275, 275]        --
|    └─MaxPool2d: 2-4                    [-1, 64, 138, 138]        --
|    └─ModuleList: 2                     []                        --
|    |    └─Sequential: 3-1              [-1, 256, 138, 138]       215,808
|    |    └─Sequential: 3-2              [-1, 512, 69, 69]         1,219,584
|    |    └─Sequential: 3-3              [-1, 1024, 35, 35]        26,090,496
|    |    └─Sequential: 3-4              [-1, 2048, 18, 18]        14,964,736
├─YolactPredictNeck: 1-2                 [-1, 256, 69, 69]         --
|    └─ModuleList: 2                     []                        --
|    |    └─Conv2d: 3-5                  [-1, 256, 18, 18]         524,544
|    |    └─Conv2d: 3-6                  [-1, 256, 35, 35]         262,400
|    |    └─Conv2d: 3-7                  [-1, 256, 69, 69]         131,328
|    └─ModuleList: 2                     []                        --
|    |    └─Conv2d: 3-8                  [-1, 256, 18, 18]         590,080
|    |    └─Conv2d: 3-9                  [-1, 256, 35, 35]         590,080
|    |    └─Conv2d: 3-10                 [-1, 256, 69, 69]         590,080
|    └─ModuleList: 2                     []                        --
|    |    └─Conv2d: 3-11                 [-1, 256, 9, 9]           590,080
|    |    └─Conv2d: 3-12                 [-1, 256, 5, 5]           590,080
├─ModuleList: 1                          []                        --
|    └─YolactPredictHead: 2-5            [[-1, 4]]                 --
|    |    └─ProtoNet: 3-13               [-1, 256, 69, 69]         590,080
|    |    └─Sequential: 3-14             [-1, 12, 69, 69]          27,660
|    |    └─Sequential: 3-15             [-1, 96, 69, 69]          221,280
|    |    └─Sequential: 3-16             [-1, 243, 69, 69]         560,115
|    └─YolactPredictHead: 2-6            [[-1, 4]]                 --
|    └─YolactPredictHead: 2              []                        --
|    |    └─ProtoNet: 3-17               [-1, 256, 35, 35]         (recursive)
|    |    └─Sequential: 3-18             [-1, 12, 35, 35]          (recursive)
|    |    └─Sequential: 3-19             [-1, 96, 35, 35]          (recursive)
|    |    └─Sequential: 3-20             [-1, 243, 35, 35]         (recursive)
|    └─YolactPredictHead: 2-7            [[-1, 4]]                 --
|    └─YolactPredictHead: 2              []                        --
|    |    └─ProtoNet: 3-21               [-1, 256, 18, 18]         (recursive)
|    |    └─Sequential: 3-22             [-1, 12, 18, 18]          (recursive)
|    |    └─Sequential: 3-23             [-1, 96, 18, 18]          (recursive)
|    |    └─Sequential: 3-24             [-1, 243, 18, 18]         (recursive)
|    └─YolactPredictHead: 2-8            [[-1, 4]]                 --
|    └─YolactPredictHead: 2              []                        --
|    |    └─ProtoNet: 3-25               [-1, 256, 9, 9]           (recursive)
|    |    └─Sequential: 3-26             [-1, 12, 9, 9]            (recursive)
|    |    └─Sequential: 3-27             [-1, 96, 9, 9]            (recursive)
|    |    └─Sequential: 3-28             [-1, 243, 9, 9]           (recursive)
|    └─YolactPredictHead: 2-9            [[-1, 4]]                 --
|    └─YolactPredictHead: 2              []                        --
|    |    └─ProtoNet: 3-29               [-1, 256, 5, 5]           (recursive)
|    |    └─Sequential: 3-30             [-1, 12, 5, 5]            (recursive)
|    |    └─Sequential: 3-31             [-1, 96, 5, 5]            (recursive)
|    |    └─Sequential: 3-32             [-1, 243, 5, 5]           (recursive)
├─ProtoNet: 1-3                          [-1, 32, 138, 138]        --
|    └─Conv2d: 2-10                      [-1, 256, 69, 69]         590,080
|    └─Conv2d: 2-11                      [-1, 256, 69, 69]         590,080
|    └─Conv2d: 2-12                      [-1, 256, 69, 69]         590,080
|    └─Upsample: 2-13                    [-1, 256, 138, 138]       --
|    └─Conv2d: 2-14                      [-1, 256, 138, 138]       590,080
|    └─Conv2d: 2-15                      [-1, 32, 138, 138]        8,224
├─Conv2d: 1-4                            [-1, 80, 69, 69]          20,560
==========================================================================================
Total params: 50,157,071
Trainable params: 50,157,071
Non-trainable params: 0
Total mult-adds (G): 34.64
==========================================================================================
Input size (MB): 3.46
Forward/backward pass size (MB): 193.40
Params size (MB): 191.33
Estimated Total Size (MB): 388.20
==========================================================================================
```