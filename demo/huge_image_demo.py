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


def parse_args():
    parser = ArgumentParser()
    parser.add_argument( 'imgs', help='Image files', nargs='+', default=[])
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
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
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--suffix', type=str, default='_result', help='suffix image result')

    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a huge image by patches
    suffix_name = args.suffix
    for im_path in args.imgs:
        print(f'Processing {im_path}')
        result = inference_detector_by_patches(model, im_path, args.patch_sizes,
                                            args.patch_steps, args.img_ratios,
                                            args.merge_iou_thr)
        # show the results
        show_result_pyplot(
            model,
            im_path,
            result,
            palette=args.palette,
            score_thr=args.score_thr,
            out_file=f'{suffix_name}'.join([ im_path.rsplit('.', 1)[0], '.jpg' ]))

if __name__ == '__main__':
    args = parse_args()
    main(args)

