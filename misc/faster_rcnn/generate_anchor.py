import math
import numpy as np

def generate_anchor(feature_height, feature_width, image_size, ratio=[0.5, 1, 2], anchor_size = [128, 256, 512]):
    #---------- debug
    assert len(image_size) == 2
    #----------
    anchor_base = []
    for ratio_t in ratio:
        for anchor_size_t in anchor_size:
            h = anchor_size_t*math.sqrt(ratio_t)
            w = anchor_size_t*math.sqrt(1/ratio_t)
            anchor_base.append([-h/2, -w/2, h/2, w/2])
    anchor_base = np.array(anchor_base) # default shape: [9,4]

    K = len(ratio) * len(anchor_size)   # default: 9
    image_height = image_size[0]
    image_width = image_size[1]
    stride_x = image_height / feature_height
    stride_y = image_width / feature_width
    anchors = np.zeros([feature_height, feature_width, K, 4])
    for i in range(feature_height):
        for j in range(feature_width):
            x = i*stride_x + stride_x/2
            y = j*stride_y + stride_y/2
            shift = [x,y,x,y]
            anchors[i, j] = anchor_base+shift

    anchors = anchors.reshape([-1,4])
    #----------
    assert isinstance(anchors, np.ndarray)
    assert anchors.shape[0] == feature_height*feature_width*len(ratio)*len(anchor_size)
    assert anchors.shape[1] == 4
    #----------
    return anchors


if __name__ == '__main__':
    anchor = generate_anchor(50, 50, (512,812.34))
    assert anchor.shape == (50*50*9,4)