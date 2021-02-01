# SSD (Single Shot MultiBox Object Detector)

## SSD Architecture

```{bash}
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─VGG: 1-1                               [-1, 64, 300, 300]        --
|    └─ModuleList: 2                     []                        --
|    |    └─Sequential: 3-1              [-1, 64, 300, 300]        38,720
|    |    └─Sequential: 3-2              [-1, 128, 150, 150]       221,440
|    |    └─Sequential: 3-3              [-1, 256, 75, 75]         1,475,328
|    |    └─Sequential: 3-4              [-1, 512, 38, 38]         5,899,776
|    |    └─Sequential: 3-5              [-1, 512, 19, 19]         7,079,424
├─SsdPredictNeck: 1-2                    [-1, 512, 38, 38]         --
|    └─L2Norm: 2-1                       [-1, 512, 38, 38]         512
|    └─ModuleList: 2                     []                        --
|    |    └─Sequential: 3-6              [-1, 1024, 19, 19]        5,769,216
|    |    └─Sequential: 3-7              [-1, 512, 10, 10]         1,442,560
|    |    └─Sequential: 3-8              [-1, 256, 5, 5]           360,832
|    |    └─Sequential: 3-9              [-1, 256, 3, 3]           328,064
|    |    └─Sequential: 3-10             [-1, 256, 1, 1]           328,064
├─ModuleList: 1                          []                        --
|    └─SsdPredictHead: 2-2               [[-1, 4]]                 --
|    |    └─Sequential: 3-11             [-1, 16, 38, 38]          73,744
|    |    └─Sequential: 3-12             [-1, 84, 38, 38]          387,156
|    └─SsdPredictHead: 2-3               [[-1, 4]]                 --
|    |    └─Sequential: 3-13             [-1, 24, 19, 19]          221,208
|    |    └─Sequential: 3-14             [-1, 126, 19, 19]         1,161,342
|    └─SsdPredictHead: 2-4               [[-1, 4]]                 --
|    |    └─Sequential: 3-15             [-1, 24, 10, 10]          110,616
|    |    └─Sequential: 3-16             [-1, 126, 10, 10]         580,734
|    └─SsdPredictHead: 2-5               [[-1, 4]]                 --
|    |    └─Sequential: 3-17             [-1, 24, 5, 5]            55,320
|    |    └─Sequential: 3-18             [-1, 126, 5, 5]           290,430
|    └─SsdPredictHead: 2-6               [[-1, 4]]                 --
|    |    └─Sequential: 3-19             [-1, 16, 3, 3]            36,880
|    |    └─Sequential: 3-20             [-1, 84, 3, 3]            193,620
|    └─SsdPredictHead: 2-7               [[-1, 4]]                 --
|    |    └─Sequential: 3-21             [-1, 16, 1, 1]            36,880
|    |    └─Sequential: 3-22             [-1, 84, 1, 1]            193,620
==========================================================================================
Total params: 26,285,486
Trainable params: 26,285,486
Non-trainable params: 0
Total mult-adds (G): 31.43
==========================================================================================
Input size (MB): 1.03
Forward/backward pass size (MB): 200.19
Params size (MB): 100.27
Estimated Total Size (MB): 301.49
```