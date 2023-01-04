import json, yaml
import numpy as np
import os, argparse
from utils.kmeans_anchors_ratios import get_optimal_anchors_ratios
from utils.kmeans_anchors_ratios import get_bboxes_adapted_to_input_size
from utils.kmeans_anchors_ratios import generate_anchors_given_ratios_and_sizes
from utils.kmeans_anchors_ratios import average_iou
from utils.kmeans_anchors_ratios import get_annotations_without_similar_anchors
import pandas as pd
import matplotlib.pyplot as plt

'''

change anchor ratios by train data size.

usage: python _04_gen_anchor_ratios.py --project {} --data_path {} --compound_coef {}

'''


def plot_bbox_area_size_agg(instances, INPUT_SIZE):
    annotations = instances["annotations"]
    image_id_larger_side_dict = {img["id"]: max(img["height"], img["width"]) for img in instances["images"]}
    bboxes_areas = [INPUT_SIZE * ann["area"] / image_id_larger_side_dict[ann["image_id"]] for ann in annotations]
    bboxes_sizes = pd.Series(np.sqrt(bboxes_areas))

    # print(bboxes_sizes.describe())

    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    bboxes_sizes.hist(bins=100)

    plt.subplot(122)
    bboxes_sizes.hist(bins=100)
    plt.xlim([0, 100])
    plt.show()
    plt.close('all')


def main(INPUT_SIZE, ANCHORS_SIZES):
    ## Get optimal anchors ratios
    anchors_ratios = get_optimal_anchors_ratios(
        instances,
        anchors_sizes=ANCHORS_SIZES,
        input_size=INPUT_SIZE,
        normalizes_bboxes=True,
        num_runs=3,
        num_anchors_ratios=3,
        max_iter=300,
        iou_threshold=0.5,
        min_size=0,
        decimals=1,
        default_anchors_ratios=[(0.7, 1.4), (1.0, 1.0), (1.4, 0.7)]
    )

    print('optimal_anchors_ratios={}'.format(anchors_ratios))

    ## Generate anchors given ratios and sizes
    anchors = generate_anchors_given_ratios_and_sizes(anchors_ratios, ANCHORS_SIZES)

    ## Get bounding boxes adapted to the input size
    resized_bboxes = get_bboxes_adapted_to_input_size(instances, input_size=INPUT_SIZE)
    resized_bboxes = resized_bboxes[resized_bboxes.prod(axis=1) > 0]  # remove 0 size

    ## Get the avg. IoU between the bounding boxes and their closest anchors
    avg_iou = average_iou(resized_bboxes, anchors)
    # print(f"Avg. IoU: {100 * avg_iou:.2f}%")

    ## Get annotations whose bounding boxes don't have similar anchors
    annotations = get_annotations_without_similar_anchors(
        instances,
        anchors_ratios,
        anchors_sizes=ANCHORS_SIZES,
        input_size=INPUT_SIZE,
        iou_threshold=0.5,
        min_size=0,
    )

    bboxes = [ann["bbox"][-2:] for ann in annotations]  # widths and heights

    instances_without_similar_anchors = instances.copy()
    instances_without_similar_anchors["annotations"] = annotations
    resized_bboxes = get_bboxes_adapted_to_input_size(instances_without_similar_anchors, input_size=INPUT_SIZE)

    # return resized_bboxes[:5]

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project', type=str, default='V01', help='project file that contains parameters')
    parser.add_argument('-d', '--data_path', type=str, default='data/PnID_updated_202212', help='target data root path')
    parser.add_argument('-c', '--compound_coef', type=int, default=4, help='model coef version default 3')
    parser.add_argument('-t', '--data_type', type=str, default='train')
    args=parser.parse_args()
    
    params = yaml.safe_load(open(f'config/202212/{args.project}/{args.project}.yml').read())
    project_name = params['project_name']
    INSTANCES_PATH = f'{args.data_path}/{project_name}/annotations/instances_{args.data_type}.json'
    instances={}
    with open(INSTANCES_PATH, encoding='utf-8') as f:
        instances = json.load(f)

    ## change the following parameters according to your model:
    # EfficientDetD{PHI}
    # PHI = 3  # for another efficientdet change only this, e.g. PHI = 3 for D3

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    # input_sizes = [1024, 1152, 1280, 1408, 1536, 1792, 1792, 2048, 2048]
    pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
    anchor_scale = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 4.0]

    scale = anchor_scale[args.compound_coef]
    strides = 2 ** np.arange(3, pyramid_levels[args.compound_coef] + 3)
    scales = np.array([0.5, 0.75, 1.0])
    # scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    INPUT_SIZE = input_sizes[args.compound_coef]
    ANCHORS_SIZES = (scale * scales * strides[:, np.newaxis]).flatten().tolist()
    # print(ANCHORS_SIZES)

    main(INPUT_SIZE, ANCHORS_SIZES)
