"""
single Inference Script of EfficientDet-Pytorch
"""
import os, argparse, json
import glob
import cv2
import random
import pandas as pd
from PIL import ImageColor


class Visualization():
    def __init__(self):
        self.class_mapping = self.__load_conf
        self.LIST = list()

    def __load_conf(self):
        with open('conf.json', 'r') as f:
            class_dict = json.load(f)
        return class_dict

    def __class_mapping(self, str):
        class_name = ''.join([self.class_mapping.get(str, 'unknown')])
        return class_name

    def __plot_one_box(self, img, coord, label=None, score=None, color=None, line_thickness=None):
        tl = line_thickness or int(round(0.001 * max(img.shape[0:2])))  # line thickness
        color = color
        c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
        cv2.rectangle(img, c1, c2, color, thickness=6)
        if label:
            tf = max(tl - 2, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0] + 15, c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            cv2.putText(img, '{}'.format(label), (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0],
                        thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)

    # Find label's index
    def __find_obj_index(self, obj_list, file_nm):
        for obj in obj_list:
            if obj in file_nm[4:]:
                return obj_list.index(obj)


    # Draw bbox on image
    def __display(self, json_nm, json, img, obj_list, save_dir_path, imshow=True, imwrite=False):
        if len(json) == 0 or len([x for i in range(len(json['objects'])) for x in json['objects'][i]['annotation']]) == 0:
            return

        file_nm = json_nm[:16]+'.png'
        obj_idx = self.__find_obj_index(obj_list, file_nm)
        label = obj_list[obj_idx]

        # color = self.color_list[obj_idx]

        os.makedirs(os.path.join(save_dir_path, 'V04_05_057'), exist_ok=True)
        for idx in range(len(json['objects'])):
            annotation = json['objects'][idx]['annotation']
            target_list = [i for i in annotation]
            if len(target_list) == 0:
                with open(os.path.join(save_dir_path, 'list.txt'), 'w', encoding='utf-8') as file:
                    file.write(file_nm.replace('.png', '.json')+'\n')
            else:
                class_label = json['objects'][idx]['class_name']

                color = annotation["meta"]
                bbox = annotation["coord"]
                x1 = bbox['x']
                y1 = bbox['y']
                x2 = bbox['x'] + bbox['width']
                y2 = bbox['y'] + bbox['height']

                self.__plot_one_box(img, [x1, y1, x2, y2], label=self.__class_mapping(class_label),
                                    color=ImageColor.getcolor(color['color'], 'RGB'))

                if imwrite:
                    cv2.imwrite(os.path.join(save_dir_path, 'V04_05_057', file_nm), img)


    # Get label name list
    def __get_obj_list(self, img_list):
        obj_list = []
        # V01_03_017_001_1.png
        for img_path in img_list:
            img_nm = os.path.basename(img_path)
            obj = img_nm[4:16]
            if obj not in obj_list:
                obj_list.append(obj)
        # print("total class count : ", len(obj_list), obj_list)

        return obj_list


    def run(self, args):

        save_dir_path = os.path.join(args.sp)

        img_list = [img for img in glob.glob(f'{args.ip}/*.png', recursive=True)]
        json_list = [json for json in glob.glob('{}/*.json'.format(args.jp), recursive=True)]
        # label / meta json list add
        obj_list = self.__get_obj_list(img_list)

        for img_path in img_list:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img_nm = os.path.basename(img_path)
            for json_path in json_list:
                json_nm = os.path.basename(json_path)
                if os.path.splitext(img_nm)[0] in json_nm:
                    json_str = ""
                    import json
                    with open(json_path, 'r', encoding='utf-8') as json_file:
                        json_str = json.load(json_file)
                        # print(json_str["objects"][0]['annotation'])
                    self.__display(json_nm, json_str, img, obj_list, save_dir_path, imshow=False, imwrite=True)


def add_arg():
    parser = argparse.ArgumentParser(description="visualize labeling data")
    parser.add_argument("--ip", required=False, type=str, help="image path", default="D:/PnID_efficientdet/SampleData/V04_TECH_Project 2022-08-05 12_16_43/images/V04_05_057")
    parser.add_argument("--jp", required=True, type=str, help="label_json path", default="D:/PnID_efficientdet/SampleData/V04_TECH_Project 2022-08-05 12_16_43/label")
    parser.add_argument("--sp", required=False, type=str, help="save path", default="D:/PnID_efficientdet/SampleData/V04_TECH_Project 2022-08-05 12_16_43/saves")
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = add_arg()
    
    v = Visualization()
    v.run(args)
