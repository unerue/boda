from .configuration_base import Config

    
config = Config({
    'backbone': Config({
        'path': 'resnet101_reducedfc.pth',
        'args': [3, 4, 23, 3]}),

    'neck': Config({
        'selected_layers': list(range(1, 4)),
        'pred_aspect_ratios': [[[1/2, 1, 2]]]*5,
        'pred_scales': [[24], [48], [96], [192], [384]],
        'num_features': 256,
        'interpolation_mode': 'bilinear',
        'num_downsample': 1,
        'use_conv_downsample': True,
        'padding': True,
        'relu_downsample_layers': False,
        'relu_pred_layers': True}),

    'head': Config({
        'mask_type': 1,
        'mask_alpha': 6.125,
        'mask_proto_src': 0,
        'mask_proto_net': [(256, 3, {'padding': 1})] * 3 + [(None, -2, {}), (256, 3, {'padding': 1})] + [(32, 1, {})],
        'mask_proto_prototypes_as_features':,
        'mask_proto_prototypes_as_features_no_grad': ,
        'mask_proto_coef_activation': ,
        'mask_proto_normalize_emulate_roi_pooling': True,
        'mask_proto_bias': ,
        'head_layer_params': {'kernel_size': 3, 'padding': 1},
        'extra_head_net': [(256, 3, {'padding': 1})],
        'extra_layers': (0, 0, 0),
        'eval_mask_branch': True,})

    'train': Config({
        'num_classes': 81,
        'max_size': 550,
        'lr_steps': (280000, 600000, 700000, 750000),
        'max_iter': 800000,}), 
})



# class YolactConfig:
#     def __init__(self):
#         self.backbone = Config({
#             'path': 'resnet101_reducedfc.pth',
#             'args': [3, 4, 23, 3]})
#         self.neck = Config({
#             'selected_layers': list(range(1, 4)),
#             'pred_aspect_ratios': [[[1/2, 1, 2]]]*5,
#             'pred_scales': [[24], [48], [96], [192], [384]],
#             'num_features': 256,
#             'interpolation_mode': 'bilinear',
#             'num_downsample': 1,
#             'use_conv_downsample': True,
#             'padding': True,
#             'relu_downsample_layers': False,
#             'relu_pred_layers': True})