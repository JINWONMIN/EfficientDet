'''
author: Min-jinwon
Date: 2022-12-09
'''

import os
import argparse
import json

import torch

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string, revision_filepath, Params

import numpy as np
import pandas as pd
import fiftyone as fo
from fiftyone import ViewField as F
from sklearn.metrics import confusion_matrix


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/202212/V02/V02.yml', required=False)
parser.add_argument('--cuda', type=boolean_string, default=True)
parser.add_argument('--float16', type=boolean_string, default=False)
parser.add_argument('--override', type=boolean_string, default=True, help='override previous bbox results file if exists')
args = parser.parse_args()

params = Params(args.config)

# eval setting
compound_coef = params.compound_coef
nms_threshold = params.nms_threshold
use_cuda = args.cuda
gpu = params.device
use_float16 = args.float16
override_prev_results = args.override
project_name = params.project_name
model_name = params.model_name

weights_path = f'weights/efficientdet-d{compound_coef}.pth' if params.weights is None else params.weights

obj_list = params.obj_list
obj_list1 = params.obj_list1

input_sizes = params.input_sizes

SET_NAME = params.test_set
VAL_GT = f'{params.ROOT}/{params.project_name}/annotations/instances_{SET_NAME}.json'
VAL_IMGS = f'{params.ROOT}/{params.project_name}/{SET_NAME}/'

# fiftyone setting
## The directory containing the source images
data_path = f'{params.ROOT}/{params.project_name}/{SET_NAME}'
## The path to the COCO style labels json file
labels_path = f'{params.ROOT}/{params.project_name}/annotations/{SET_NAME}.json'
## The type of the dataset being imported
dataset_type = fo.types.COCODetectionDataset

def run(Dataset):
    dataset = Dataset
    dataset.persistent = True
    dataset.overwrite = True
    
    # Get class list
    classes = dataset.default_classes
    
    print(f'running coco-style evaluation on project {project_name}, weights {weights_path}...')
   
    if override_prev_results or not os.path.exists(f'{project_name}_{SET_NAME}_bbox_results.json'):
        model = EfficientDetBackbone(compound_coef=compound_coef,
                                     num_classes=len(obj_list),
                                     ratios=eval(params.anchors_ratios),
                                     scales=eval(params.anchors_scales))
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.requires_grad_(False)
        model.eval()
        
        if use_cuda:
            model.cuda(gpu)
            
            if use_float16:
                model.half()
                
        with fo.ProgressBar() as pb:
            for sample in pb(dataset):
                ori_imgs, framed_imgs, framed_metas = preprocess(sample.filepath,
                                                                 max_size=input_sizes[compound_coef],
                                                                 mean=params.mean,
                                                                 std=params.std)

                _, h, w, c = np.array(ori_imgs).shape
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

                # Perform inference
                features, regression, classification, anchors = model(x)

                threshold = 0.2
                regressBoxes = BBoxTransform()
                clipBoxes = ClipBoxes()
                
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
                detections = []
                if rois.shape[0] > 0:
                    # convert to [top-left-x, top-left-y, width, height]
                    rois[:, 2] -= rois[:, 0]
                    rois[:, 3] -= rois[:, 1]

                    bbox_score = scores
                    
                    for roi_id in range(rois.shape[0]):
                        score = float(bbox_score[roi_id])
                        label = int(class_ids[roi_id])
                        box = rois[roi_id, :]
                        x1, y1, x2, y2 = box
                        rel_box = [x1 / w, y1 / h, x2 / w, y2 / h]
                        detections.append(
                            fo.Detection(
                                label=classes[label + 1],
                                bounding_box=rel_box,
                                confidence=float(score)
                            )
                        )
                        
                # Save predictions to dataset
                sample['predictions'] = fo.Detections(detections=detections)
                sample.save()
                
        # Evaluate the predictions in the 'EfficientDet-d4-V04' field of out 'high_conf_view'
        # with respect to the objects in the 'ground_truth' field
        results = dataset.evaluate_detections(
            "predictions",
            gt_field="ground_truth_detections",
            eval_key="eval",
            classwise=False,
            iou=params.iou,
            compute_mAP=True,
            iou_threshs=params.iou_threshs
        )
        
        print(f'mAP: {results.mAP()}')
        plot = results.plot_pr_curves(classes=obj_list)
        plot.write_image(f'result/{project_name}/{model_name}/PR_curve.png', format='png', width=1500, height=1200)
        plot.show()
        
        plot1= results.plot_confusion_matrix(classes=obj_list,
                                            colorscale='edge',
                                            log_colorscale=True)
        plot1.save(f'result/{project_name}/{model_name}/Confusion_Matrix.png', format='png')
        plot1.show()
        
        results.write_json(f'result/{project_name}/{model_name}/{model_name}_d{compound_coef}_results.json')
        
    return results        
                
                
if __name__ == "__main__":
    revision_filepath(file_path=VAL_GT,
                      ROOT=params.ROOT,
                      project_name=params.project_name,
                      SET_NAME=SET_NAME)
    
    
    dataset = fo.Dataset.from_dir(
        dataset_type=dataset_type,
        labels_path=labels_path,
        label_field="ground_truth",
        # name=model_name
    )
    
    results = run(dataset)

    with open(f'result/{project_name}/{model_name}/{model_name}_d{compound_coef}_results.json', 'r', encoding='utf-8') as coco:
        json_ = json.load(coco)

    y_true = json_['ytrue']
    y_pred = json_['ypred']

    con = confusion_matrix(y_true, y_pred)
    con_df = pd.DataFrame(con, columns=obj_list1, index=obj_list1)
    con_df.to_csv(f'result/{project_name}/{model_name}/{model_name}_d{compound_coef}_results.csv', encoding='utf-8')

    # Print a classification report 
    results.print_report(classes=obj_list)

    # Print some statistics about the total TP/FP/FN counts
    print("TP: %d" % dataset.sum("eval_tp"))
    print("FP: %d" % dataset.sum("eval_fp"))
    print("FN: %d" % dataset.sum("eval_fn"))

    # Create a view that has samples with the most false positives first, and
    # only includes false positive boxes in the 'predictions' field
    view = (
        dataset
        .sort_by("eval_fp", reverse=True)
        .filter_labels("predictions", F("eval") == "fp")
    )

    # Visualize results in the App
    # session = fo.launch_app(view=view, remote=True)
    
