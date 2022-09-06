import torch
from torch.backends import cudnn

from backbone import EfficientDetBackbone
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time, datetime

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess
import argparse
import yaml, json
import os, glob

'''

Get inferenced result by the weights.

usage: python _06_infer_test.py --project {} --compound_coef {} --weights {} --data_path {} 

'''


ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='V01', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=3, help='coefficients of efficientdet')
ap.add_argument('-w', '--weights', type=str, default='result/V01_DSME_Project 2022-08-03 14_54_01_resize/20220820_200520/efficientdet-d4_200_1005_best.pth', help='/path/to/weights')
ap.add_argument('-d', '--data_path', type=str, default='data/PnID', help='/path/to/data')
ap.add_argument('--iou_threshold', type=float, default=0.5, help='nms threshold(IOU), don\'t change it if not for testing purposes')

args = ap.parse_args()

compound_coef = args.compound_coef
force_input_size = None  # set None to use default size

threshold = 0.1      ## confidence threshold
iou_threshold = args.iou_threshold

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True
gpu = 1

params = yaml.safe_load(open(f'config/{args.project}.yml').read())
#params = yaml.safe_load(open(f'config/V01_resize.yml').read())
project_name = params['project_name']
obj_list = params['obj_list']
SET_NAME = params['train_set']
VAL_GT = f'{args.data_path}/{params["project_name"]}/annotations/instances_{SET_NAME}.json'


# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

weight_path = args.weights
test_path = f'{args.data_path}/{params["project_name"]}/{SET_NAME}'
#weight_path = 'result/V01_DSME_Project 2022-08-03 14_54_01_resize/20220818_143003/efficientdet-d4_2828_14145_best.pth'
#test_path = f'data/PnID_220818/V01_DSME_Project 2022-08-03 14_54_01_resize/val'
test_list = glob.glob('{}/*.*'.format(test_path))

now = datetime.datetime.now().strftime('%Y%m%d%H%M')
save_img_path = f'{os.path.dirname(args.weights)}/infer_{now}'
#save_img_path = f'result/V01_DSME_Project 2022-08-03 14_54_01_resize/20220818_143003/infer_{now}'
os.makedirs(save_img_path, exist_ok=True)

for img_path in test_list:
    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda(gpu) for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
                                # replace this part with your project's anchor config
                                #ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                                #scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cuda')))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda(gpu)
    if use_float16:
        model = model.half()

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)

    out = invert_affine(framed_metas, out)

    for i in range(len(ori_imgs)):
        if len(out[i]['rois']) == 0:
            continue
        ori_imgs[i] = ori_imgs[i].copy()
        for j in range(len(out[i]['rois'])):
            (x1, y1, x2, y2) = out[i]['rois'][j].astype(np.int)
            cv2.rectangle(ori_imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[out[i]['class_ids'][j]]
            score = float(out[i]['scores'][j])

            cv2.putText(ori_imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)

            plt.imshow(ori_imgs[i])

        cv2.imwrite(os.path.join(save_img_path, os.path.basename(img_path)), ori_imgs[i])
        print(f'>>>>> completed save image -> {os.path.join(save_img_path, os.path.basename(img_path))}')
