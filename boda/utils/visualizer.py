import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch


def draw_masks(masks):
    return masks


def draw_boxes(boxes):
    return boxes


def get_color(j, on_gpu=None):
    global color_cache
    color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
    
    if on_gpu is not None and color_idx in color_cache[on_gpu]:
        return color_cache[on_gpu][color_idx]
    else:
        color = COLORS[color_idx]
        if not undo_transform:
            # The image might come in as RGB or BRG, depending
            color = (color[2], color[1], color[0])
        if on_gpu is not None:
            color = torch.Tensor(color).to(on_gpu).float() / 255.
            color_cache[on_gpu][color_idx] = color
        return color


class Visualizer:
    def __init__(self):
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)

    def restore_transformed_image(self, img, w, h):
        """
        Takes a transformed image tensor and returns a numpy ndarray that is untransformed.
        Arguments w and h are the original height and width of the image.

        Args:
            size (:obj:`Tuple[int]`)
        """
        img_numpy = img.permute(1, 2, 0).cpu().numpy()
        img_numpy = img_numpy[:, :, (2, 1, 0)]

        # if cfg.backbone.transform.normalize:
        img_numpy = (img_numpy * np.array(self.std) + np.array(self.mean)) / 255.0
        # elif cfg.backbone.transform.subtract_means:
        #     img_numpy = (img_numpy / 255.0 + np.array(MEANS) / 255.0).astype(np.float32)

        img_numpy = img_numpy[:, :, (2, 1, 0)]
        img_numpy = np.clip(img_numpy, 0, 1)

        return cv2.resize(img_numpy, (w, h))

    def prep_display(
        self,
        dets_out,
        img,
        h,
        w,
        undo_transform=True,
        class_color=False,
        mask_alpha=0.45,
        fps_str=''
    ) -> None:
        if undo_transform:
            img_numpy = self.restore_transformed_image(img, w, h)
            img_gpu = torch.Tensor(img_numpy).cuda()
        else:
            img_gpu = img / 255.0
            h, w, _ = img.shape

        # First drawing bounding boxes
        # Lastly masks        

        idx = t[1].argsort(0, descending=True)[:args.top_k]

        masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

        num_dets_to_consider = min(args.top_k, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < args.score_threshold:
                num_dets_to_consider = j
                break

        if args.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
            # After this, mask is of size [num_dets, h, w, 1]
            masks = masks[:num_dets_to_consider, :, :, None]
            
            # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
            colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
            masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

            # This is 1 everywhere except for 1-mask_alpha where the mask is
            inv_alph_masks = masks * (-mask_alpha) + 1
            
            # I did the math for this on pen and paper. This whole block should be equivalent to:
            #    for j in range(num_dets_to_consider):
            #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
            masks_color_summand = masks_color[0]
            if num_dets_to_consider > 1:
                inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
                masks_color_cumul = masks_color[1:] * inv_alph_cumul
                masks_color_summand += masks_color_cumul.sum(dim=0)

            img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
        
        if args.display_fps:
                # Draw the box for the fps on the GPU
            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            font_thickness = 1

            text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

            img_gpu[0:text_h+8, 0:text_w+8] *= 0.6 # 1 - Box alpha


        # Then draw the stuff that needs to be done on the cpu
        # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
        img_numpy = (img_gpu * 255).byte().cpu().numpy()

        if args.display_fps:
            # Draw the text on the CPU
            text_pt = (4, text_h + 2)
            text_color = [255, 255, 255]

            cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
        if num_dets_to_consider == 0:
            return img_numpy

        if args.display_text or args.display_bboxes:
            for j in reversed(range(num_dets_to_consider)):
                x1, y1, x2, y2 = boxes[j, :]
                color = get_color(j)
                score = scores[j]

                if args.display_bboxes:
                    cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

                if args.display_text:
                    _class = cfg.dataset.class_names[classes[j]]
                    text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class

                    font_face = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.6
                    font_thickness = 1

                    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                    text_pt = (x1, y1 - 3)
                    text_color = [255, 255, 255]

                    cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                    cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

        return img_numpy




