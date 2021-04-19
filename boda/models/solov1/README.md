# SOLO (Segmenting Objects by Locations)

```
 ██████╗  ██████╗ ██╗      ██████╗         
██╔════╝ ██╔═══██╗██║     ██╔═══██╗        
╚██████╗ ██║   ██║██║     ██║   ██║██╗   ██╗
 ╚════██╗██║   ██║██║     ██║   ██║ ██╗ ██╔╝
 ██████╔╝╚██████╔╝███████╗╚██████╔╝  ████╔╝ 
 ╚═════╝  ╚═════╝ ╚══════╝ ╚═════╝   ╚═══╝  
```

## SOLO Architecture

```{bash}
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─ResNet: 1-1                            [-1, 256, 334, 200]       --
|    └─Conv2d: 2-1                       [-1, 64, 667, 400]        9,408
|    └─BatchNorm2d: 2-2                  [-1, 64, 667, 400]        128
|    └─ReLU: 2-3                         [-1, 64, 667, 400]        --
|    └─MaxPool2d: 2-4                    [-1, 64, 334, 200]        --
|    └─ModuleList: 2                     []                        --
|    |    └─Sequential: 3-1              [-1, 256, 334, 200]       215,808
|    |    └─Sequential: 3-2              [-1, 512, 167, 100]       1,219,584
|    |    └─Sequential: 3-3              [-1, 1024, 84, 50]        7,098,368
|    |    └─Sequential: 3-4              [-1, 2048, 42, 25]        14,964,736
├─Solov1PredictNeck: 1-2                 [-1, 256, 334, 200]       --
|    └─ModuleList: 2                     []                        --
|    |    └─Conv2d: 3-5                  [-1, 256, 42, 25]         524,544
|    |    └─Conv2d: 3-6                  [-1, 256, 84, 50]         262,400
|    |    └─Conv2d: 3-7                  [-1, 256, 167, 100]       131,328
|    |    └─Conv2d: 3-8                  [-1, 256, 334, 200]       65,792
|    └─ModuleList: 2                     []                        --
|    |    └─Conv2d: 3-9                  [-1, 256, 42, 25]         590,080
|    |    └─Conv2d: 3-10                 [-1, 256, 84, 50]         590,080
|    |    └─Conv2d: 3-11                 [-1, 256, 167, 100]       590,080
|    |    └─Conv2d: 3-12                 [-1, 256, 334, 200]       590,080
|    └─ModuleList: 2                     []                        --
|    |    └─Conv2d: 3-13                 [-1, 256, 21, 13]         590,080
├─Solov1PredictHead: 1-3                 [2, 1600, 334, 200]       --
|    └─ModuleList: 2                     []                        --
|    |    └─Sequential: 3-14             [-1, 256, 167, 100]       595,200
|    |    └─Sequential: 3-15             [-1, 256, 167, 100]       590,592
|    |    └─Sequential: 3-16             [-1, 256, 167, 100]       590,592
|    |    └─Sequential: 3-17             [-1, 256, 167, 100]       590,592
|    |    └─Sequential: 3-18             [-1, 256, 167, 100]       590,592
|    |    └─Sequential: 3-19             [-1, 256, 167, 100]       590,592
|    |    └─Sequential: 3-20             [-1, 256, 167, 100]       590,592
|    └─ModuleList: 2                     []                        --
|    |    └─Conv2d: 3-21                 [-1, 1600, 334, 200]      411,200
|    └─ModuleList: 2                     []                        --
|    |    └─Sequential: 3-22             [-1, 256, 40, 40]         590,592
|    |    └─Sequential: 3-23             [-1, 256, 40, 40]         590,592
|    |    └─Sequential: 3-24             [-1, 256, 40, 40]         590,592
|    |    └─Sequential: 3-25             [-1, 256, 40, 40]         590,592
|    |    └─Sequential: 3-26             [-1, 256, 40, 40]         590,592
|    |    └─Sequential: 3-27             [-1, 256, 40, 40]         590,592
|    |    └─Sequential: 3-28             [-1, 256, 40, 40]         590,592
|    └─Conv2d: 2-5                       [-1, 79, 40, 40]          182,095
|    └─ModuleList: 2                     []                        --
|    |    └─Sequential: 3-29             [-1, 256, 167, 100]       (recursive)
|    |    └─Sequential: 3-30             [-1, 256, 167, 100]       (recursive)
|    |    └─Sequential: 3-31             [-1, 256, 167, 100]       (recursive)
|    |    └─Sequential: 3-32             [-1, 256, 167, 100]       (recursive)
|    |    └─Sequential: 3-33             [-1, 256, 167, 100]       (recursive)
|    |    └─Sequential: 3-34             [-1, 256, 167, 100]       (recursive)
|    |    └─Sequential: 3-35             [-1, 256, 167, 100]       (recursive)
|    └─ModuleList: 2                     []                        --
|    |    └─Conv2d: 3-36                 [-1, 1296, 334, 200]      333,072
|    └─ModuleList: 2                     []                        --
|    |    └─Sequential: 3-37             [-1, 256, 36, 36]         (recursive)
|    |    └─Sequential: 3-38             [-1, 256, 36, 36]         (recursive)
|    |    └─Sequential: 3-39             [-1, 256, 36, 36]         (recursive)
|    |    └─Sequential: 3-40             [-1, 256, 36, 36]         (recursive)
|    |    └─Sequential: 3-41             [-1, 256, 36, 36]         (recursive)
|    |    └─Sequential: 3-42             [-1, 256, 36, 36]         (recursive)
|    |    └─Sequential: 3-43             [-1, 256, 36, 36]         (recursive)
|    └─Conv2d: 2-6                       [-1, 79, 36, 36]          (recursive)
|    └─ModuleList: 2                     []                        --
|    |    └─Sequential: 3-44             [-1, 256, 84, 50]         (recursive)
|    |    └─Sequential: 3-45             [-1, 256, 84, 50]         (recursive)
|    |    └─Sequential: 3-46             [-1, 256, 84, 50]         (recursive)
|    |    └─Sequential: 3-47             [-1, 256, 84, 50]         (recursive)
|    |    └─Sequential: 3-48             [-1, 256, 84, 50]         (recursive)
|    |    └─Sequential: 3-49             [-1, 256, 84, 50]         (recursive)
|    |    └─Sequential: 3-50             [-1, 256, 84, 50]         (recursive)
|    └─ModuleList: 2                     []                        --
|    |    └─Conv2d: 3-51                 [-1, 576, 168, 100]       148,032
|    └─ModuleList: 2                     []                        --
|    |    └─Sequential: 3-52             [-1, 256, 24, 24]         (recursive)
|    |    └─Sequential: 3-53             [-1, 256, 24, 24]         (recursive)
|    |    └─Sequential: 3-54             [-1, 256, 24, 24]         (recursive)
|    |    └─Sequential: 3-55             [-1, 256, 24, 24]         (recursive)
|    |    └─Sequential: 3-56             [-1, 256, 24, 24]         (recursive)
|    |    └─Sequential: 3-57             [-1, 256, 24, 24]         (recursive)
|    |    └─Sequential: 3-58             [-1, 256, 24, 24]         (recursive)
|    └─Conv2d: 2-7                       [-1, 79, 24, 24]          (recursive)
|    └─ModuleList: 2                     []                        --
|    |    └─Sequential: 3-59             [-1, 256, 42, 25]         (recursive)
|    |    └─Sequential: 3-60             [-1, 256, 42, 25]         (recursive)
|    |    └─Sequential: 3-61             [-1, 256, 42, 25]         (recursive)
|    |    └─Sequential: 3-62             [-1, 256, 42, 25]         (recursive)
|    |    └─Sequential: 3-63             [-1, 256, 42, 25]         (recursive)
|    |    └─Sequential: 3-64             [-1, 256, 42, 25]         (recursive)
|    |    └─Sequential: 3-65             [-1, 256, 42, 25]         (recursive)
|    └─ModuleList: 2                     []                        --
|    |    └─Conv2d: 3-66                 [-1, 256, 84, 50]         65,792
|    └─ModuleList: 2                     []                        --
|    |    └─Sequential: 3-67             [-1, 256, 16, 16]         (recursive)
|    |    └─Sequential: 3-68             [-1, 256, 16, 16]         (recursive)
|    |    └─Sequential: 3-69             [-1, 256, 16, 16]         (recursive)
|    |    └─Sequential: 3-70             [-1, 256, 16, 16]         (recursive)
|    |    └─Sequential: 3-71             [-1, 256, 16, 16]         (recursive)
|    |    └─Sequential: 3-72             [-1, 256, 16, 16]         (recursive)
|    |    └─Sequential: 3-73             [-1, 256, 16, 16]         (recursive)
|    └─Conv2d: 2-8                       [-1, 79, 16, 16]          (recursive)
|    └─ModuleList: 2                     []                        --
|    |    └─Sequential: 3-74             [-1, 256, 42, 25]         (recursive)
|    |    └─Sequential: 3-75             [-1, 256, 42, 25]         (recursive)
|    |    └─Sequential: 3-76             [-1, 256, 42, 25]         (recursive)
|    |    └─Sequential: 3-77             [-1, 256, 42, 25]         (recursive)
|    |    └─Sequential: 3-78             [-1, 256, 42, 25]         (recursive)
|    |    └─Sequential: 3-79             [-1, 256, 42, 25]         (recursive)
|    |    └─Sequential: 3-80             [-1, 256, 42, 25]         (recursive)
|    └─ModuleList: 2                     []                        --
|    |    └─Conv2d: 3-81                 [-1, 144, 84, 50]         37,008
|    └─ModuleList: 2                     []                        --
|    |    └─Sequential: 3-82             [-1, 256, 12, 12]         (recursive)
|    |    └─Sequential: 3-83             [-1, 256, 12, 12]         (recursive)
|    |    └─Sequential: 3-84             [-1, 256, 12, 12]         (recursive)
|    |    └─Sequential: 3-85             [-1, 256, 12, 12]         (recursive)
|    |    └─Sequential: 3-86             [-1, 256, 12, 12]         (recursive)
|    |    └─Sequential: 3-87             [-1, 256, 12, 12]         (recursive)
|    |    └─Sequential: 3-88             [-1, 256, 12, 12]         (recursive)
|    └─Conv2d: 2-9                       [-1, 79, 12, 12]          (recursive)
==========================================================================================
Total params: 36,892,591
Trainable params: 36,892,591
Non-trainable params: 0
Total mult-adds (G): 296.58
==========================================================================================
Input size (MB): 12.20
Forward/backward pass size (MB): 2671.69
Params size (MB): 140.73
Estimated Total Size (MB): 2824.63
==========================================================================================
```