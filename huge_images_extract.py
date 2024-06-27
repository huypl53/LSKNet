# Copyright (c) OpenMMLab. All rights reserved.
"""Inference on huge images.

Example:
```
wget -P checkpoint https://download.openmmlab.com/mmrotate/v0.1.0/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth  # noqa: E501, E261.
python demo/huge_image_demo.py \
    demo/dota_demo.jpg \
    configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_v3.py \
    checkpoint/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth \
```
"""  # nowq

from argparse import ArgumentParser

from mmdet.apis import init_detector, show_result_pyplot

from mmrotate.apis import inference_detector_by_patches
from rotated_rect_crop import crop_rotated_rectangle
import math

import glob
import numpy as np
import os
import cv2
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--img', help='Image file')
    group.add_argument('--dir', help='Directory contains files')

    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--patch_sizes',
        type=int,
        nargs='+',
        default=[1024],
        help='The sizes of patches')
    parser.add_argument(
        '--patch_steps',
        type=int,
        nargs='+',
        default=[824],
        help='The steps between two patches')
    parser.add_argument(
        '--img_ratios',
        type=float,
        nargs='+',
        default=[1.0],
        help='Image resizing ratios for multi-scale detecting')
    parser.add_argument(
        '--merge_iou_thr',
        type=float,
        default=0.1,
        help='IoU threshould for merging results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    # parser.add_argument(
    #     '--palette',
    #     default='dota',
    #     choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'],
    #     help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--save-dir', type=str, default='./cropped_patches', help='directory to store cropped patches')
    args = parser.parse_args()
    return args


def crop_rotated_boxes(image, bboxes) -> np.ndarray:
    return [crop_rotated_rectangle(image, box) for box in bboxes]


def get_cropped_boxes(im_path: str, im, model, args):
    result = inference_detector_by_patches(model, im_path, args.patch_sizes,
                                           args.patch_steps, args.img_ratios,
                                           args.merge_iou_thr)

    if not len(result) or not len(result[0]):
        return

    # im = cv2.imread(im_path)
    rbboxes = [[(int(r[0]), int(r[1])), (int(r[2]), int(r[3])), math.degrees(r[4])]
               for r in result[0] if r[-1] > args.score_thr]

    return crop_rotated_boxes(im, rbboxes)


def save_cropped_patches(im_path, out_dir,  cropped_patches):
    bname = os.path.basename(im_path)
    splited_bnames = bname.rsplit('.', 1)
    try:
        for i, patch in enumerate(cropped_patches):
            out_name = f'{i:03}.'.join(splited_bnames)
            out_path = os.path.join(out_dir, out_name)
            cv2.imwrite(out_path, patch)
    except Exception as e:
        print(e)
    pass


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a huge image by patches

    if args.img:
        im_path = args.img
        im = cv2.imread(im_path)
        cropped_patches = get_cropped_boxes(im_path, im, model, args)
        save_cropped_patches(im_path, cropped_patches)
        return

    im_paths = glob.glob(f'{args.dir}/*.*')

    os.makedirs(args.save_dir, exist_ok=True)
    for i, im_path in enumerate(tqdm(im_paths)):
        im = cv2.imread(im_path)
        cropped_patches = get_cropped_boxes(im_path, im, model, args)
        save_cropped_patches(im_path, args.save_dir, cropped_patches)


'''
python ./demo/huge_images_extract.py --dir ../images  --config './configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py' \
--checkpoint '../epoch_3_050324.pth' \
--score-thr 0.5 \
--save-dir ./ships/
'''

if __name__ == '__main__':
    args = parse_args()
    main(args)
