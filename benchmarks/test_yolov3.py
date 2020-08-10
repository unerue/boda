import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from backbone import Darknet
from backbone import darknet53
from architecture import Yolov3Model, Yolov3PredictionHead
from configuration import yolov3_base_darknet_pascal
from architecture import Yolov3Loss


if __name__ == '__main__':
    from torchsummary import summary

    # config_path = '/home/unerue/Documents/computer-vision/detection/backbone/yolov3.cfg'
    # # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # # print(device)
    # backbone = Darknet(config_path)
    # print(summary(backbone, input_data=(3, 416, 416), verbose=0))

    # cnt = 0
    # for i in backbone.config:
    #     if i['type'] == 'convolutional':
    #         cnt += 1
    #         print(cnt, 'convolutional')
    #     elif i['type'] == 'backbone':
    #         print('='*50)
    #     elif i['type'] == 'upsample':
    #         cnt += 1
    #         print(cnt, 'upsample*********')

    #     elif i['type'] == 'route':
    #         cnt += 1
    #         print(cnt, 'route**********', i['layers'])

    #     elif i['type'] == 'shortcut':
    #         cnt += 1
    #         print(cnt, 'shortcut')

    #     elif i['type'] == 'yolo':
    #         cnt += 1
    #         print(cnt, 'yolo')


    backbone = darknet53()
    print(summary(backbone, input_data=(3, 416, 416), verbose=0))

    yolov3 = Yolov3PredictionHead(yolov3_base_darknet_pascal)
    print(summary(yolov3, input_data=(3, 416, 416), verbose=0))

    yolov3 = Yolov3Model(yolov3_base_darknet_pascal)
    print(summary(yolov3, input_data=(3, 416, 416), verbose=0))

    device = 'cuda' if torch.cuda.is_available() else 'cpu' # 삭제해라!!!!
    test_tensor = torch.ByteTensor().to(device)
    print(test_tensor.is_cuda)



    model = Yolov3Model(yolov3_base_darknet_pascal).to(device)
    criterion = Yolov3Loss()
    images = []
    for i in range(3):
        images.append(torch.randn((3, 3, 416, 416)))

    targets = []    
    
    for j in range(3):
        target = {
            'boxes': torch.randn((3, 10, 4)),
            'labels': torch.randint(1, 21, (3, 10, 1))
        }
        targets.append(target)

    print(len(images), len(targets))

    import sys
    # sys.exit(0)
    for image, target in zip(images, targets):
        inputs = image.to(device)
        
        outputs = model(inputs)
        print(outputs[0]['boxes'][0][0])
        print(outputs[0]['scores'][0][0])
        print(outputs[0]['labels'][0][0][0])
        criterion(outputs, target)



    
    
    # test_yolov3 = TestYolov3(config_path)
    # print(summary(test_yolov3, input_data=(3, 416, 416), verbose=0))

    # # print(summary(backbone, input_data=(3, 416, 416), verbose=0))
    # # print(backbone.channels)
    # # print(len(backbone.channels))
    # # print(len(backbone.backbone_modules))

    # cnt = 0
    # for i in test_yolov3.config:
    #     if i['type'] == 'convolutional':
    #         cnt += 1
    #         print(cnt, 'convolutional')
    #     elif i['type'] == 'backbone':
    #         print('='*50)
    #     elif i['type'] == 'upsample':
    #         cnt += 1
    #         print(cnt, 'upsample*********')

    #     elif i['type'] == 'route':
    #         cnt += 1
    #         print(cnt, 'route**********', i['layers'])

    #     elif i['type'] == 'shortcut':
    #         cnt += 1
    #         print(cnt, 'shortcut')

    #     elif i['type'] == 'yolo':
    #         cnt += 1
    #         print(cnt, 'yolo')

    # # print(len(backbone.backbone_modules))
    # # print(backbone.modules)

    # yolov3 = Yolov3(backbone, config_path)

    # print(summary(yolov3, input_data=(3, 416, 416), verbose=0, device='cuda'))

    # # cnt = 0
    # # for i in backbone.config:
    # #     if i['type'] == 'convolutional':
    # #         cnt += 1
    # #         print(cnt, 'convolutional')
    # #     elif i['type'] == 'backbone':
    # #         print('='*50)
    # #     elif i['type'] == 'upsample':
    # #         cnt += 1
    # #         print(cnt, 'upsample')

    # #     elif i['type'] == 'route':
    # #         cnt += 1
    # #         print(cnt, 'route**********')

    # #     elif i['type'] == 'shortcut':
    # #         cnt += 1
    # #         print(cnt, 'shortcut')

    # #     elif i['type'] == 'yolo':
    # #         cnt += 1
    # #         print(cnt, 'yolo')

    # # print(backbone.backbone_modules[61])
    # # print(len(backbone.backbone_modules))