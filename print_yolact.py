import torch


state_dict = torch.load('resnet50_550_coco_972_70000.pth')
# print(state_dict.keys())

for key, value in state_dict.items():
    if key[:8] != 'backbone':
        print(key, value.size())

state_dict = torch.load('yolact.pth')
# print(state_dict.keys())
print()
for key, value in state_dict.items():
    if key[:8] != 'backbone':
        print(key, value.size())





# fpn.lat_layers.0.weight torch.Size([256, 2048, 1, 1])
# fpn.lat_layers.0.bias torch.Size([256])
# fpn.lat_layers.1.weight torch.Size([256, 1024, 1, 1])
# fpn.lat_layers.1.bias torch.Size([256])
# fpn.lat_layers.2.weight torch.Size([256, 512, 1, 1])
# fpn.lat_layers.2.bias torch.Size([256])
# fpn.pred_layers.0.weight torch.Size([256, 256, 3, 3])
# fpn.pred_layers.0.bias torch.Size([256])
# fpn.pred_layers.1.weight torch.Size([256, 256, 3, 3])
# fpn.pred_layers.1.bias torch.Size([256])
# fpn.pred_layers.2.weight torch.Size([256, 256, 3, 3])
# fpn.pred_layers.2.bias torch.Size([256])
# fpn.downsample_layers.0.weight torch.Size([256, 256, 3, 3])
# fpn.downsample_layers.0.bias torch.Size([256])
# fpn.downsample_layers.1.weight torch.Size([256, 256, 3, 3])
# fpn.downsample_layers.1.bias torch.Size([256])

# proto_net.0.weight torch.Size([256, 256, 3, 3])
# proto_net.0.bias torch.Size([256])
# proto_net.2.weight torch.Size([256, 256, 3, 3])
# proto_net.2.bias torch.Size([256])
# proto_net.4.weight torch.Size([256, 256, 3, 3])
# proto_net.4.bias torch.Size([256])
# proto_net.8.weight torch.Size([256, 256, 3, 3])
# proto_net.8.bias torch.Size([256])
# proto_net.10.weight torch.Size([32, 256, 1, 1])
# proto_net.10.bias torch.Size([32])

# prediction_layers.0.upfeature.0.weight torch.Size([256, 256, 3, 3])
# prediction_layers.0.upfeature.0.bias torch.Size([256])
# prediction_layers.0.bbox_layer.weight torch.Size([20, 256, 3, 3])
# prediction_layers.0.bbox_layer.bias torch.Size([20])
# prediction_layers.0.conf_layer.weight torch.Size([35, 256, 3, 3])
# prediction_layers.0.conf_layer.bias torch.Size([35])
# prediction_layers.0.mask_layer.weight torch.Size([160, 256, 3, 3])
# prediction_layers.0.mask_layer.bias torch.Size([160])
# semantic_seg_conv.weight torch.Size([6, 256, 1, 1])
# semantic_seg_conv.bias torch.Size([6])


# neck.lateral_layers.0.weight torch.Size([256, 2048, 1, 1])
# neck.lateral_layers.0.bias torch.Size([256])
# neck.lateral_layers.1.weight torch.Size([256, 1024, 1, 1])
# neck.lateral_layers.1.bias torch.Size([256])
# neck.lateral_layers.2.weight torch.Size([256, 512, 1, 1])
# neck.lateral_layers.2.bias torch.Size([256])
# neck.predict_layers.0.weight torch.Size([256, 256, 3, 3])
# neck.predict_layers.0.bias torch.Size([256])
# neck.predict_layers.1.weight torch.Size([256, 256, 3, 3])
# neck.predict_layers.1.bias torch.Size([256])
# neck.predict_layers.2.weight torch.Size([256, 256, 3, 3])
# neck.predict_layers.2.bias torch.Size([256])
# neck.downsample_layers.0.weight torch.Size([256, 256, 3, 3])
# neck.downsample_layers.0.bias torch.Size([256])
# neck.downsample_layers.1.weight torch.Size([256, 256, 3, 3])
# neck.downsample_layers.1.bias torch.Size([256])

# proto_net.protonet1.weight torch.Size([256, 256, 3, 3])
# proto_net.protonet1.bias torch.Size([256])
# proto_net.protonet2.weight torch.Size([256, 256, 3, 3])
# proto_net.protonet2.bias torch.Size([256])
# proto_net.protonet3.weight torch.Size([256, 256, 3, 3])
# proto_net.protonet3.bias torch.Size([256])
# proto_net.protonet5.weight torch.Size([256, 256, 3, 3])
# proto_net.protonet5.bias torch.Size([256])
# proto_net.protonet6.weight torch.Size([32, 256, 1, 1])
# proto_net.protonet6.bias torch.Size([32])