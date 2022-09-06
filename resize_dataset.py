import argparse
import json
import os
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox
from collections import defaultdict
import sys
import math
from tqdm import tqdm


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    format_str = "{0:." + str(decimals) + "f}"
    percent = format_str.format(100 * (iteration / float(total)))
    filled_length = int(math.ceil(bar_length * iteration / float(total)))
    bar = '>' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write('\r%s |%s| %s %s %s%s %s' % (prefix, bar, str(iteration) + " / " + str(total), "|", percent, '%', suffix))

    if iteration == total:
        sys.stdout.write('\n')

    sys.stdout.flush()

def resize_image_and_bounding_boxes(imgFile, bboxes, inputW, inputH, targetImgW, targetImgH, outputImgFile):
    print("Reading image {0}".format(imgFile))
    img = cv2.imread(imgFile)

    if inputW > inputH:
        seq = iaa.Sequential([
            iaa.Resize({"height": "keep-aspect-ratio", "width": targetImgW}),
            iaa.PadToFixedSize(width=targetImgW, height=targetImgH)
        ])
    else:
        seq = iaa.Sequential([
            iaa.Resize({"height": targetImgH, "width": "keep-aspect-ratio"}),
            iaa.PadToFixedSize(width=targetImgW, height=targetImgH)
        ])

    image_aug, bbs_aug = seq(image=img, bounding_boxes=bboxes)

    print("Writing resized image {0}".format(outputImgFile))
    cv2.imwrite(outputImgFile, image_aug)
    print("Resized image {0} written successfully".format(outputImgFile))

    return bbs_aug

def run(args, curr_cnt=int(0)):
    imageDir = args['images_dir']
    annotationsFile = args['annotations_file']
    targetImgW = int(args['image_width'])
    targetImgH = int(args['image_height'])
    outputImageDir = args['output_img_dir']
    outputAnnotationsFile = args['output_ann_file']

    print("Loading annotations file")
    data = json.load(open(annotationsFile, 'r'))
    print("Annotations file loaded.")

    print("Building dictionnaries")
    anns = defaultdict(list)
    annsIdx = dict()
    for i in range(0, len(data['annotations'])):
        anns[data['annotations'][i]['image_id']].append(data['annotations'][i])
        annsIdx[data['annotations'][i]['id']] = i
    print("Dictionnaries built.")

    for img in data['images']:
        curr_cnt += 1
        print("Processing image file {0} and its bounding boxes".format(img['file_name']))

        annList = anns[img['id']]

        bboxesList = []
        for ann in tqdm(annList):
            bboxData = ann['bbox']
            bboxesList.append(
                BoundingBox(x1=bboxData[0], y1=bboxData[1], x2=bboxData[2], y2=bboxData[3]))

        imgFullPath = os.path.join(imageDir, img['file_name'])
        outputImgFullPath = os.path.join(outputImageDir, img['file_name'])
        outputDir = os.path.dirname(outputImgFullPath)

        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        if not os.path.exists(outputAnnotationsFile):
            os.makedirs(outputAnnotationsFile)

        outNewBBoxes = resize_image_and_bounding_boxes(imgFullPath, bboxesList, int(img['width']), int(img['height']),
                                                       targetImgW, targetImgH, outputImgFullPath)

        for i in range(0, len(annList)):
            annId = annList[i]['id']
            data['annotations'][annsIdx[annId]]['bbox'][0] = outNewBBoxes[i].x1
            data['annotations'][annsIdx[annId]]['bbox'][1] = outNewBBoxes[i].y1
            data['annotations'][annsIdx[annId]]['bbox'][2] = outNewBBoxes[i].x2
            data['annotations'][annsIdx[annId]]['bbox'][3] = outNewBBoxes[i].y2

        img['width'] = targetImgW
        img['height'] = targetImgH

        print_progress(curr_cnt, len(data['images']), 'Progress:', 'Complete', 1, len(data['images']))

    print('\n')
    print('images cnt: ', curr_cnt, '\t', 'anootations cnt: ', len(data['annotations']))
    print('\n')

    print("#################### JSON DUMP ####################")
    print('\n')
    print("Writing modified annotations to file...")
    with open(os.path.join(outputAnnotationsFile, f'instances_{mode}.json'), 'w') as outfile:
        json.dump(data, outfile, indent=4)
    print('\n')
    print('Finish')


if __name__ == "__main__":
    '''
    project_type =  "V01_DSME_Project 2022-08-03 14_54_01"
                    "V02_AKER_Project 2022-08-05 12_17_05" 
                    "V03_KBR_Project 2022-08-03 14_56_42" 
                    "V04_TECH_Project 2022-08-05 12_16_43" 
    '''
    project_type = "V01_DSME_Project 2022-08-03 14_54_01"
    mode = "train"  # mode = ["train", "test", "val"]

    ia.seed(1)

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--images_dir", required=False,
                        default=f"D:/PnID_efficientdet/data/PnID/{project_type}/{mode}",
                        help="Directory where are located the images referenced in the annotations file")
    parser.add_argument("-a", "--annotations_file", required=False,
                        default=f"D:/PnID_efficientdet/data/PnID/{project_type}/annotations/instances_{mode}.json",
                        help="COCO JSON format annotations file")
    parser.add_argument("-w", "--image_width", default=1024, required=False, help="Target image width")
    parser.add_argument("-t", "--image_height", default=1024, required=False, help="Target image height")
    parser.add_argument("-o", "--output_ann_file", required=False,
                        default=f"D:/PnID_efficientdet/data/PnID_resize/{project_type}_resize/annotations",
                        help="Output annotations file")
    parser.add_argument("-f", "--output_img_dir", required=False,
                        default=f"D:/PnID_efficientdet/data/PnID_resize/{project_type}_resize/{mode}",
                        help="Output images directory")

    args = vars(parser.parse_args())

    run(args)
