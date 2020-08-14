from configuration_base import Config
from configuration_base import PASCAL_CLASSES, COCO_CLASSES, COCO_LABEL_MAP


dataset_base = Config({
    'name': 'Pascal SBD 2012',

    'train_images': './data',
    'valid_images': './data',
    
    'train_labels': './data/',
    'valid_labels': './data/',

    'class_names': PASCAL_CLASSES,
    'num_classes': 20,
})

backbone_base = Config({
    'name': 'Base Backbone',
    'path': 'path/to/pretrained/weights',
})

yolov1_base = Config({
    'name': 'Base Backbone',
    'path': 'path/to/pretrained/weights',

})

darknet9_backbone = backbone_base.copy({
    'name': 'ResNet50',
    'path': 'resnet50-19c8e357.pth',
})

coco_base_config = Config({
    'dataset': coco2014_dataset,
    'num_classes': 81, # This should include the background class

    'max_iter': 400000,

    # The maximum number of detections for evaluation
    'max_num_detections': 100,

    # dw' = momentum * dw - lr * (grad + decay * w)
    'lr': 1e-3,
    'momentum': 0.9,
    'decay': 5e-4,

    # For each lr step, what to multiply the lr with
    'gamma': 0.1,
    'lr_steps': (280000, 360000, 400000),

    # Initial learning rate to linearly warmup from (if until > 0)
    'lr_warmup_init': 1e-4,

    # If > 0 then increase the lr linearly from warmup_init to lr each iter for until iters
    'lr_warmup_until': 500,

    # The terms to scale the respective loss by
    'conf_alpha': 1,
    'bbox_alpha': 1.5,
    'mask_alpha': 0.4 / 256 * 140 * 140, # Some funky equation. Don't worry about it.

    # Eval.py sets this if you just want to run YOLACT as a detector
    'eval_mask_branch': True,

    # Top_k examples to consider for NMS
    'nms_top_k': 200,
    # Examples with confidence less than this are not considered by NMS
    'nms_conf_thresh': 0.05,
    # Boxes with IoU overlap greater than this threshold will be culled during NMS
    'nms_thresh': 0.5,

    # See mask_type for details.
    'mask_type': mask_type.direct,
    'mask_size': 16,
    'masks_to_train': 100,
    'mask_proto_src': None,
    'mask_proto_net': [(256, 3, {}), (256, 3, {})],
    'mask_proto_bias': False,
    'mask_proto_prototype_activation': activation_func.relu,
    'mask_proto_mask_activation': activation_func.sigmoid,
    'mask_proto_coeff_activation': activation_func.tanh,
    'mask_proto_crop': True,
    'mask_proto_crop_expand': 0,
    'mask_proto_loss': None,
    'mask_proto_binarize_downsampled_gt': True,
    'mask_proto_normalize_mask_loss_by_sqrt_area': False,
    'mask_proto_reweight_mask_loss': False,
    'mask_proto_grid_file': 'data/grid.npy',
    'mask_proto_use_grid':  False,
    'mask_proto_coeff_gate': False,
    'mask_proto_prototypes_as_features': False,
    'mask_proto_prototypes_as_features_no_grad': False,
    'mask_proto_remove_empty_masks': False,
    'mask_proto_reweight_coeff': 1,
    'mask_proto_coeff_diversity_loss': False,
    'mask_proto_coeff_diversity_alpha': 1,
    'mask_proto_normalize_emulate_roi_pooling': False,
    'mask_proto_double_loss': False,
    'mask_proto_double_loss_alpha': 1,
    'mask_proto_split_prototypes_by_head': False,
    'mask_proto_crop_with_pred_box': False,

    # SSD data augmentation parameters
    # Randomize hue, vibrance, etc.
    'augment_photometric_distort': True,
    # Have a chance to scale down the image and pad (to emulate smaller detections)
    'augment_expand': True,
    # Potentialy sample a random crop from the image and put it in a random place
    'augment_random_sample_crop': True,
    # Mirror the image with a probability of 1/2
    'augment_random_mirror': True,
    # Flip the image vertically with a probability of 1/2
    'augment_random_flip': False,
    # With uniform probability, rotate the image [0,90,180,270] degrees
    'augment_random_rot90': False,

    # Discard detections with width and height smaller than this (in absolute width and height)
    'discard_box_width': 4 / 550,
    'discard_box_height': 4 / 550,

    # If using batchnorm anywhere in the backbone, freeze the batchnorm layer during training.
    # Note: any additional batch norm layers after the backbone will not be frozen.
    'freeze_bn': False,

    # Set this to a config object if you want an FPN (inherit from fpn_base). See fpn_base for details.
    'fpn': None,

    # Use the same weights for each network head
    'share_prediction_module': False,

    # For hard negative mining, instead of using the negatives that are leastl confidently background,
    # use negatives that are most confidently not background.
    'ohem_use_most_confident': False,

    # Use focal loss as described in https://arxiv.org/pdf/1708.02002.pdf instead of OHEM
    'use_focal_loss': False,
    'focal_loss_alpha': 0.25,
    'focal_loss_gamma': 2,
    
    # The initial bias toward forground objects, as specified in the focal loss paper
    'focal_loss_init_pi': 0.01,

    # Keeps track of the average number of examples for each class, and weights the loss for that class accordingly.
    'use_class_balanced_conf': False,

    # Whether to use sigmoid focal loss instead of softmax, all else being the same.
    'use_sigmoid_focal_loss': False,

    # Use class[0] to be the objectness score and class[1:] to be the softmax predicted class.
    # Note: at the moment this is only implemented if use_focal_loss is on.
    'use_objectness_score': False,

    # Adds a global pool + fc layer to the smallest selected layer that predicts the existence of each of the 80 classes.
    # This branch is only evaluated during training time and is just there for multitask learning.
    'use_class_existence_loss': False,
    'class_existence_alpha': 1,

    # Adds a 1x1 convolution directly to the biggest selected layer that predicts a semantic segmentations for each of the 80 classes.
    # This branch is only evaluated during training time and is just there for multitask learning.
    'use_semantic_segmentation_loss': False,
    'semantic_segmentation_alpha': 1,

    # Adds another branch to the netwok to predict Mask IoU.
    'use_mask_scoring': False,
    'mask_scoring_alpha': 1,

    # Match gt boxes using the Box2Pix change metric instead of the standard IoU metric.
    # Note that the threshold you set for iou_threshold should be negative with this setting on.
    'use_change_matching': False,

    # Uses the same network format as mask_proto_net, except this time it's for adding extra head layers before the final
    # prediction in prediction modules. If this is none, no extra layers will be added.
    'extra_head_net': None,

    # What params should the final head layers have (the ones that predict box, confidence, and mask coeffs)
    'head_layer_params': {'kernel_size': 3, 'padding': 1},

    # Add extra layers between the backbone and the network heads
    # The order is (bbox, conf, mask)
    'extra_layers': (0, 0, 0),

    # During training, to match detections with gt, first compute the maximum gt IoU for each prior.
    # Then, any of those priors whose maximum overlap is over the positive threshold, mark as positive.
    # For any priors whose maximum is less than the negative iou threshold, mark them as negative.
    # The rest are neutral and not used in calculating the loss.
    'positive_iou_threshold': 0.5,
    'negative_iou_threshold': 0.5,

    # When using ohem, the ratio between positives and negatives (3 means 3 negatives to 1 positive)
    'ohem_negpos_ratio': 3,

    # If less than 1, anchors treated as a negative that have a crowd iou over this threshold with
    # the crowd boxes will be treated as a neutral.
    'crowd_iou_threshold': 1,

    # This is filled in at runtime by Yolact's __init__, so don't touch it
    'mask_dim': None,

    # Input image size.
    'max_size': 300,
    
    # Whether or not to do post processing on the cpu at test time
    'force_cpu_nms': True,

    # Whether to use mask coefficient cosine similarity nms instead of bbox iou nms
    'use_coeff_nms': False,

    # Whether or not to have a separate branch whose sole purpose is to act as the coefficients for coeff_diversity_loss
    # Remember to turn on coeff_diversity_loss, or these extra coefficients won't do anything!
    # To see their effect, also remember to turn on use_coeff_nms.
    'use_instance_coeff': False,
    'num_instance_coeffs': 64,

    # Whether or not to tie the mask loss / box loss to 0
    'train_masks': True,
    'train_boxes': True,
    # If enabled, the gt masks will be cropped using the gt bboxes instead of the predicted ones.
    # This speeds up training time considerably but results in much worse mAP at test time.
    'use_gt_bboxes': False,

    # Whether or not to preserve aspect ratio when resizing the image.
    # If True, this will resize all images to be max_size^2 pixels in area while keeping aspect ratio.
    # If False, all images are resized to max_size x max_size
    'preserve_aspect_ratio': False,

    # Whether or not to use the prediction module (c) from DSSD
    'use_prediction_module': False,

    # Whether or not to use the predicted coordinate scheme from Yolo v2
    'use_yolo_regressors': False,
    
    # For training, bboxes are considered "positive" if their anchors have a 0.5 IoU overlap
    # or greater with a ground truth box. If this is true, instead of using the anchor boxes
    # for this IoU computation, the matching function will use the predicted bbox coordinates.
    # Don't turn this on if you're not using yolo regressors!
    'use_prediction_matching': False,

    # A list of settings to apply after the specified iteration. Each element of the list should look like
    # (iteration, config_dict) where config_dict is a dictionary you'd pass into a config object's init.
    'delayed_settings': [],

    # Use command-line arguments to set this.
    'no_jit': False,

    'backbone': None,
    'name': 'base_config',

    # Fast Mask Re-scoring Network
    # Inspried by Mask Scoring R-CNN (https://arxiv.org/abs/1903.00241)
    # Do not crop out the mask with bbox but slide a convnet on the image-size mask,
    # then use global pooling to get the final mask score
    'use_maskiou': False,
    
    # Archecture for the mask iou network. A (num_classes-1, 1, {}) layer is appended to the end.
    'maskiou_net': [],

    # Discard predicted masks whose area is less than this
    'discard_mask_area': -1,

    'maskiou_alpha': 1.0,
    'rescore_mask': False,
    'rescore_bbox': False,
    'maskious_to_train': -1,
})


# ----------------------- YOLACT v1.0 CONFIGS ----------------------- #

yolact_base_config = coco_base_config.copy({
    'name': 'yolact_base',

    # Dataset stuff
    'dataset': coco2017_dataset,
    'num_classes': len(coco2017_dataset.class_names) + 1,

    # Image Size
    'max_size': 550,
    
    # Training params
    'lr_steps': (280000, 600000, 700000, 750000),
    'max_iter': 800000,
    
    # Backbone Settings
    'backbone': resnet101_backbone.copy({
        'selected_layers': list(range(1, 4)),
        'use_pixel_scales': True,
        'preapply_sqrt': False,
        'use_square_anchors': True, # This is for backward compatability with a bug

        'pred_aspect_ratios': [ [[1, 1/2, 2]] ]*5,
        'pred_scales': [[24], [48], [96], [192], [384]],
    }),

    # FPN Settings
    'fpn': fpn_base.copy({
        'use_conv_downsample': True,
        'num_downsample': 2,
    }),

    # Mask Settings
    'mask_type': mask_type.lincomb,
    'mask_alpha': 6.125,
    'mask_proto_src': 0,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 3 + [(None, -2, {}), (256, 3, {'padding': 1})] + [(32, 1, {})],
    'mask_proto_normalize_emulate_roi_pooling': True,

    # Other stuff
    'share_prediction_module': True,
    'extra_head_net': [(256, 3, {'padding': 1})],

    'positive_iou_threshold': 0.5,
    'negative_iou_threshold': 0.4,

    'crowd_iou_threshold': 0.7,

    'use_semantic_segmentation_loss': True,
})


yolact_resnet50_config = yolact_base_config.copy({
    'name': 'yolact_resnet50',

    'backbone': resnet50_backbone.copy({
        'selected_layers': list(range(1, 4)),
        
        'pred_scales': yolact_base_config.backbone.pred_scales,
        'pred_aspect_ratios': yolact_base_config.backbone.pred_aspect_ratios,
        'use_pixel_scales': True,
        'preapply_sqrt': False,
        'use_square_anchors': True, # This is for backward compatability with a bug
    }),
})


yolact_resnet50_pascal_config = yolact_resnet50_config.copy({
    'name': None, # Will default to yolact_resnet50_pascal
    
    # Dataset stuff
    'dataset': pascal_dataset,
    'num_classes': len(pascal_dataset.class_names) + 1,

    'max_iter': 120000,
    'lr_steps': (60000, 100000),
    
    'backbone': yolact_resnet50_config.backbone.copy({
        'pred_scales': [[32], [64], [128], [256], [512]],
        'use_square_anchors': False,
    })
})


if __name__ == '__main__':
    print(pascal_dataset.print())
    print(resnet50_backbone.print())