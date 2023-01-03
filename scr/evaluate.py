# Author: Zylo117
'''
author: Min-jinwon
Date: 2022-12-09
'''

"""
COCO-Style Evaluations
put images here datasets/your_project_name/val_set_name/*.jpg
put annotations here datasets/your_project_name/annotations/instances_{val_set_name}.json
put weights here /path/to/your/weights/*.pth
change compound_coef
"""
'''
author: Min-jinwon
revision version
Date: 2022-12-09
'''

import json
import os

import argparse
import torch
import cv2
import random
import numpy as np

import shutil

from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string, plot_one_box, standard_to_bgr, STANDARD_COLORS, Params

from torchvision.ops.boxes import batched_nms
from torchsummaryX import summary


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/202212/V01/V01.yml')
parser.add_argument('--cuda', type=boolean_string, default=True)
parser.add_argument('--float16', type=boolean_string, default=False)
parser.add_argument('--override', type=boolean_string, default=True, help='override previous bbox results file if exists')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--display-bboxes', dest='display_bboxes', action='store_true', help='Bounding box display in soruce images')
args = parser.parse_args()

params = Params(args.config)

compound_coef = params.compound_coef
nms_threshold = params.nms_threshold
use_cuda = args.cuda
gpu = params.device
use_float16 = args.float16
override_prev_results = args.override
project_name = params.project_name
threshold = params.threshold
root = params.ROOT

weights_path = f'weights/efficientdet-d{compound_coef}.pth' if params.weights is None else params.weights

print(f'running coco-style evaluation on project {project_name}, weights {weights_path}...')

obj_list = params.obj_list

color_list = standard_to_bgr(STANDARD_COLORS)

input_sizes = params.input_sizes
input_size = input_sizes[compound_coef]

if not os.path.isdir(params.result_path):
    os.mkdir(params.result_path)
    print(params.result_path + ' created')
if not os.path.isdir(params.result_path + '/fail'):
    os.mkdir(params.result_path + '/fail')
    print(params.result_path + '/fail created')

def display(preds, imgs, img, ori_imgs, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            cv2.imwrite(f'{params.result_path}/fail/{img}', imgs[i])
            shutil.copy(f'{ori_imgs}/{img}', f'{params.result_path}/{img}.jpg')
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            # plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=random.choice(color_list))

        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            cv2.imwrite(f'{params.result_path}/{img}', imgs[i])
            shutil.copy(f'{ori_imgs}/{img}', f'{params.result_path}/{img}.jpg')


def evaluate_coco(img_path, set_name, image_ids, coco, model, Dataset, threshold=0.2):
    results = []
    
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    
    for image_id in tqdm(image_ids, total=len(image_ids), desc='test dataset evaluation', leave=True):
        image_info = coco.loadImgs(image_id)[0]
        image_path = img_path + image_info['file_name']

        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, 
                                                         max_size=input_sizes[compound_coef],
                                                         mean=params.mean,
                                                         std=params.std)

        x = torch.from_numpy(framed_imgs[0])

        if use_cuda:
            x = x.cuda(gpu)
            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()

        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        features, regression, classification, anchors = model(x)

        preds = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, nms_threshold)
        
        if not preds:
            continue

        preds = invert_affine(framed_metas, preds)[0]
        
        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']
        
        # Convert detections to FiftyOne format

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_id': image_id,
                    'category_id': label + 1,
                    'score': float(score),
                    'bbox': box.tolist(),
                    'img_name': image_info['file_name'],
                    'slice_bbox': image_info['slices'],
                    'ori_file_name': image_info['ori_file_name']
                }

                results.append(image_result)
    
    if not len(results):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')

    # write output
    filepath = f'{project_name}_{set_name}_bbox_results.json'
    if os.path.exists(filepath):
        os.remove(filepath)
        
    json.dump(results, open(filepath, 'w'), indent=4)

def _eval(coco_gt, image_ids, pred_json_path):\
    
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')

    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    SET_NAME = params.test_set
    VAL_GT = f'{root}/{params.project_name}/annotations/instances_{SET_NAME}.json'
    VAL_IMGS = f'{root}/{params.project_name}/{SET_NAME}/'
    # ORI_IMGS = f'{root}/visualization/{params.visual_path}/{params.project_name}/{SET_NAME}'
    ORI_IMGS = f'{root}/{params.project_name}/ori_split/{SET_NAME}'

    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()[:]

    print('==============================================================================================================================')
    print('\t \t \t \t \t \t \t load model')
    
    if args.evaluate:
        if override_prev_results or not os.path.exists(f'{project_name}_{SET_NAME}_bbox_results.json'):
            model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                        ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
            summary(model, torch.rand((1, 3, input_size, input_size)))
            
            print('\t \t \t \t \t \t \t start evaluation')
            print(f'\t \t \t \t \t \t \t confidence score threshold: {threshold}')
            
            model.requires_grad_(False)
            model.eval()

            if use_cuda:
                model.cuda(gpu)

                if use_float16:
                    model.half()

            evaluate_coco(VAL_IMGS, SET_NAME, image_ids, coco_gt, model, threshold)

    _eval(coco_gt, image_ids, f'{project_name}_{SET_NAME}_bbox_results.json')
    
    if args.display_bboxes:
        with open(f'./{project_name}_{SET_NAME}_bbox_results.json', 'r', encoding='utf-8') as f:
            preds = json.load(f)
            
        ori_file_name = set([x['ori_file_name'] for x in preds])
        
        print('==============================================================================================================================')
        print('\t \t \t \t \t \t \t displaying...')

        for name in tqdm(ori_file_name, total=len(ori_file_name), desc='prediction result display', leave=True):
            bboxes = []
            scores = []
            cate_id = []
            out = []
            
            print(name)
            ori_imgs, framed_imgs, framed_metas = preprocess(os.path.join(ORI_IMGS, name), max_size=input_size)
            
            ori_name_lookup = list(filter(lambda x: x['ori_file_name'].endswith(name), preds))
            
            for i in range(len(ori_name_lookup)):
                xymin = [ori_name_lookup[i]['bbox'][0] + ori_name_lookup[i]['slice_bbox'][0], 
                        ori_name_lookup[i]['bbox'][1] + ori_name_lookup[i]['slice_bbox'][1]]
                ori_name_lookup[i]['cvt_bbox'] = [ori_name_lookup[i]['bbox'][0] + ori_name_lookup[i]['slice_bbox'][0],
                                                ori_name_lookup[i]['bbox'][1] + ori_name_lookup[i]['slice_bbox'][1],
                                                ori_name_lookup[i]['bbox'][2] + xymin[0],
                                                ori_name_lookup[i]['bbox'][3] + xymin[1]]
                bboxes.append(ori_name_lookup[i]['cvt_bbox'])
                scores.append(ori_name_lookup[i]['score'])
                cate_id.append(ori_name_lookup[i]['category_id']-1)
                
            bboxes = torch.from_numpy(np.array(bboxes))
            scores = torch.from_numpy(np.array(scores))
            cate_id = torch.from_numpy(np.array(cate_id))

            bboxes_after_nms = batched_nms(bboxes, scores, cate_id, nms_threshold)
            if bboxes_after_nms.shape[0] != 0:
                classes_ = cate_id[bboxes_after_nms]
                scores_ = scores[bboxes_after_nms]
                bboxes_ = bboxes[bboxes_after_nms]

                out.append({
                    'rois': bboxes_.numpy(),
                    'class_ids': classes_.numpy(),
                    'scores': scores_.numpy(),
                })
            else:
                out.append({
                    'rois': np.array(()),
                    'class_ids': np.array(()),
                    'scores': np.array(()),
                })
                
            display(out, ori_imgs, name, ORI_IMGS, imshow=False, imwrite=True)
        
    print('Finish...!')
    
