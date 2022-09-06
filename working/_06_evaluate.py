import os
import cv2 as cv
import numpy as np
from collections import Counter
import json, yaml
import datetime
import argparse
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string
from operator import itemgetter


'''

Get evaluated result by the weights.

usage: python _06_evaluate.py --project {} --compound_coef {} --weights {} --data_path {} 

'''


ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='V01', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=3, help='coefficients of efficientdet')
ap.add_argument('-w', '--weights', type=str, default='result/V01_DSME_Project 2022-08-03 14_54_01_resize/20220820_200520/efficientdet-d4_200_1005_best.pth', help='/path/to/weights')
ap.add_argument('-d', '--data_path', type=str, default='data/PnID', help='/path/to/data')
ap.add_argument('--nms_threshold', type=float, default=0.5, help='nms threshold(IOU), don\'t change it if not for testing purposes')
ap.add_argument('--cuda', type=boolean_string, default=True)
ap.add_argument('--device', type=int, default=1)
ap.add_argument('--float16', type=boolean_string, default=False)
ap.add_argument('--override', type=boolean_string, default=True, help='override previous bbox results file if exists')
args = ap.parse_args()

compound_coef = args.compound_coef
nms_threshold = args.nms_threshold
use_cuda = args.cuda
gpu = args.device
use_float16 = args.float16
override_prev_results = args.override

weights_path = f'weights/efficientdet-d{compound_coef}.pth' if args.weights is None else args.weights
print(f'running coco-style evaluation on project {args.project}, weights {weights_path}...')

params = yaml.safe_load(open(f'config/{args.project}.yml'))
project_name = params['project_name']
obj_list = params['obj_list']

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
num2class = {}

def convert_to_pascal_voc_coord(coord_tp):
    # input: coco -> return: pascal VOC
    x1 = round(coord_tp[0])
    y1 = round(coord_tp[1])
    x2 = round(coord_tp[0] + coord_tp[2])
    y2 = round(coord_tp[1] + coord_tp[3])
    return (x1, y1, x2, y2)


def revert_coco_coord(coord_tp):
    # input: pascal VOC -> return coco
    x1 = coord_tp[0]
    y1 = coord_tp[1]
    w = coord_tp[2] - coord_tp[0]
    h = coord_tp[3] - coord_tp[1]
    return (x1, y1, w, h)


def convertToAbsoluteValues(size, box):
    xIn = round(((2 * float(box[0]) - float(box[2])) * size[0] / 2))
    yIn = round(((2 * float(box[1]) - float(box[3])) * size[1] / 2))
    xEnd = xIn + round(float(box[2]) * size[0])
    yEnd = yIn + round(float(box[3]) * size[1])

    if xIn < 0:
        xIn = 0
    if yIn < 0:
        yIn = 0
    if xEnd >= size[0]:
        xEnd = size[0] - 1
    if yEnd >= size[1]:
        yEnd = size[1] - 1
    return (xIn, yIn, xEnd, yEnd)


def evaluate_coco(img_path, image_ids, coco, model, save_json_path, threshold=0.05):
    results = []

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path = img_path + image_info['file_name']

        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef],
                                                         mean=params['mean'], std=params['std'])
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

        if rois.shape[0] > 0:
            # # x1,y1,x2,y2 -> x1,y1,w,h
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
                    'file_name': image_info['file_name']
                }
                

                results.append(image_result)

    if not len(results):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')

    if os.path.exists(save_json_path):
        os.remove(save_json_path)
    json.dump(results, open(save_json_path, 'w'), indent=4)
    return results

def evaluate_coco_custom(img_path, image_ids, coco, model, save_json_path, threshold=0.05):
    results = []

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    print(f'len(image_ids)={len(image_ids)}')
    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path = img_path + image_info['file_name']

        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef],
                                                         mean=params['mean'], std=params['std'])
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

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            results_one_img = []
            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_id': image_id,
                    'category_id': label + 1,
                    'score': float(score),
                    'bbox': box.tolist(),
                    'file_name': image_info['file_name']
                }

                results_one_img.append(image_result)

	    
            # 1개의 이미지에서 maximum score 객체만 전체 결과에 반영
            # results_one_img = sorted(results_one_img, key=itemgetter('score'), reverse=True)
            results.append(results_one_img)

    print(f'len(results)={len(results)}')
    if not len(results):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')

    # write output
    # filepath = f'{set_name}_bbox_results.json'
    if os.path.exists(save_json_path):
        os.remove(save_json_path)
    json.dump(results, open(save_json_path, 'w'), indent=4)
    return results

def load_GT_DT(dt_path, anno_path, DT=None):
    with open(anno_path, 'r', encoding='utf-8') as jf:
        GT = json.load(jf)

    if DT == None:
        with open(dt_path, 'r', encoding='utf-8') as jf:
            DT = json.load(jf)

    detections, groundtruths, classes = [], [], []
    for ct in GT['categories']:
        classes.append(ct['id'])
        num2class[str(ct['id'])]=ct['name']

    for idx, img_dict in enumerate(DT):
        coord_tp = convert_to_pascal_voc_coord(img_dict['bbox'])
        boxinfo = [img_dict['file_name'], img_dict['category_id'], img_dict['score'], coord_tp]
        detections.append(boxinfo)

    # GT
    for idx, img_dict in enumerate(GT['images']):
        anno_dict = list(filter(lambda ori_anno_dict: ori_anno_dict['image_id'] == img_dict['id'], GT['annotations']))[0]
        coord_tp = convert_to_pascal_voc_coord(anno_dict['bbox'])
        # coord_tp = convertToAbsoluteValues((img_dict['width'], img_dict['height']), anno_dict['bbox'])
        boxinfo = [img_dict['file_name'], anno_dict['category_id'], 1.0, coord_tp]
        groundtruths.append(boxinfo)

    return detections, groundtruths, classes


def boxPlot(boxlist, imagePath, savePath):
    imgfiles = sorted(list(set([filename for filename, _, _, _ in boxlist])))

    for img_file in imgfiles:
        rect_infos = []
        imgfile_path = os.path.join(imagePath, img_file)
        img = cv.imread(imgfile_path)
        for filename, _, conf, (x1, y1, x2, y2) in boxlist:
            if img_file == filename:
                rect_infos.append((x1, y1, x2, y2, conf))
        for x1, y1, x2, y2, conf in rect_infos:
            if conf == 1.0:
                rectcolor = (0, 255, 0)
            else:
                rectcolor = (0, 0, 255)
            cv.rectangle(img, (x1, y1), (x2, y2), rectcolor, 5)
        cv.imwrite(f"{savePath}/{img_file}", img)



def getArea(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def getUnionAreas(boxA, boxB, interArea=None):
    area_A = getArea(boxA)
    area_B = getArea(boxB)

    if interArea is None:
        interArea = getIntersectionArea(boxA, boxB)

    return float(area_A + area_B - interArea)


def getIntersectionArea(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # intersection area
    return (xB - xA + 1) * (yB - yA + 1)


def boxesIntersect(boxA, boxB):
    if boxA[0] > boxB[2]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[2]:
        return False  # boxA is left of boxB
    if boxA[3] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[3]:
        return False  # boxA is below boxB
    return True


def iou(boxA, boxB):
    # if boxes dont intersect
    if boxesIntersect(boxA, boxB) is False:
        return 0
    interArea = getIntersectionArea(boxA, boxB)
    union = getUnionAreas(boxA, boxB, interArea=interArea)

    # intersection over union
    result = interArea / union
    assert result >= 0
    return result



######### AP
def calculateAveragePrecision(rec, prec):
    mrec = [0] + [e for e in rec] + [1]
    mpre = [0] + [e for e in prec] + [0]

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    ii = []

    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)

    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])

    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]


def ElevenPointInterpolatedAP(rec, prec):

    mrec = [e for e in rec]
    mpre = [e for e in prec]

    recallValues = np.linspace(0, 1, 11)
    recallValues = list(recallValues[::-1])
    rhoInterp, recallValid = [], []

    for r in recallValues:
        argGreaterRecalls = np.argwhere(mrec[:] >= r)
        pmax = 0

        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min():])

        recallValid.append(r)
        rhoInterp.append(pmax)

    ap = sum(rhoInterp) / 11

    return [ap, rhoInterp, recallValues, None]


def AP(detections, groundtruths, classes, IOUThreshold=0.5, method='AP'):
    result = []

    for c in classes:

        dects = [d for d in detections if d[1] == c]
        gts = [g for g in groundtruths if g[1] == c]

        npos = len(gts)

        dects = sorted(dects, key=lambda conf: conf[2], reverse=True)

        TP = np.zeros(len(dects))
        FP = np.zeros(len(dects))

        det = Counter(cc[0] for cc in gts)

        for key, val in det.items():
            det[key] = np.zeros(val)

        for d in range(len(dects)):

            gt = [gt for gt in gts if gt[0] == dects[d][0]]

            iouMax = 0

            for j in range(len(gt)):
                iou1 = iou(dects[d][3], gt[j][3])
                if iou1 > iouMax:
                    iouMax = iou1
                    jmax = j

            if iouMax >= IOUThreshold:
                if det[dects[d][0]][jmax] == 0:
                    TP[d] = 1
                    det[dects[d][0]][jmax] = 1
                else:
                    FP[d] = 1
            else:
                FP[d] = 1

        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        rec = acc_TP / npos
        prec = np.divide(acc_TP, (acc_FP + acc_TP))

        if method == "AP":
            [ap, mpre, mrec, ii] = calculateAveragePrecision(rec, prec)
        else:
            [ap, mpre, mrec, _] = ElevenPointInterpolatedAP(rec, prec)

        r = {
            'class': c,
            'precision': prec,
            'recall': rec,
            'AP': ap,
            'interpolated precision': mpre,
            'interpolated recall': mrec,
            'total positives': npos,
            'total TP': np.sum(TP),
            'total FP': np.sum(FP)
        }

        result.append(r)

    return result

####### mAP
def mAP(result):
    ap = 0
    for r in result:
        ap += r['AP']
    mAP = ap / len(result)

    return mAP





if __name__=='__main__':
    SET_NAME = params['val_set']
    VAL_GT = f'{args.data_path}/{params["project_name"]}/annotations/instances_{SET_NAME}.json'
    VAL_IMGS = f'{args.data_path}/{params["project_name"]}/{SET_NAME}/'
    coco_gt = COCO(VAL_GT)

    image_ids = sorted(coco_gt.getImgIds())
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_json_path = f'{os.path.dirname(args.weights)}/{now}_{SET_NAME}_bbox_results.json'

    if override_prev_results or not os.path.exists(save_json_path):
        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                     ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.requires_grad_(False)
        model.eval()

        if use_cuda:
            model.cuda(gpu)

            if use_float16:
                model.half()

        result = evaluate_coco(VAL_IMGS, image_ids, coco_gt, model, save_json_path)
        detections, groundtruths, classes = load_GT_DT(save_json_path, VAL_GT, DT=result)

        result = AP(detections, groundtruths, classes)
        print(len(result))

        for r in result:
            print("{:^8} AP : {}".format(num2class[str(int(r['class']))], r['AP']))
        print("---------------------------")
        print(f"mAP : {mAP(result)}")
