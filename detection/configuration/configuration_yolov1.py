class Yolov3Config:
    def __init__(self):
        
        pass

"""
Pascal VOC or MS COCO 
[x_min, y_min, x_max, y_max]
[x, y, w, h]


"""

"""
image: a PIL Image of size (H, W)
target: a dict containing the following fields
boxes (FloatTensor[N, 4]): the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, 
                            ranging from 0 to W and 0 to H
labels (Int64Tensor[N]): the label for each bounding box. 0 represents always the background class.
image_id (Int64Tensor[1]): an image identifier. It should be unique between all the images in the dataset, 
                           and is used during evaluation
area (Tensor[N]): The area of the bounding box. 
                  This is used during evaluation with the COCO metric, 
                  to separate the metric scores between small, medium and large boxes.

iscrowd (UInt8Tensor[N]): instances with iscrowd=True will be ignored during evaluation.
# (optionally) masks (UInt8Tensor[N, H, W]): The segmentation masks for each one of the objects
# (optionally) keypoints (FloatTensor[N, K, 3]): For each one of the N objects, 
# it contains the K keypoints in [x, y, visibility] format, defining the object. visibility=0 means that the keypoint is not visible. Note that for data augmentation, the notion of flipping a keypoint is dependent on the data representation, and you should probably adapt references/detection/transforms.py for your new keypoint representation
"""