import os, re, json, argparse, yaml
from _ctypes import PyObj_FromPtr
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import tqdm


'''

resize images 

usage: python _03_resize_image.py --project {} --data_path {} --img_size {}

'''


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

class NoIndent(object):
    def __init__(self, value):
        self.value = value

class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

    def __init__(self, **kwargs):
        self.__sort_keys = kwargs.get('sort_keys', None)
        super(MyEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent) else super(MyEncoder, self).default(obj))

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC
        json_repr = super(MyEncoder, self).encode(obj)

        for match in self.regex.finditer(json_repr):
            id = int(match.group(1))
            no_indent = PyObj_FromPtr(id)
            json_obj_repr = json.dumps(no_indent.value, sort_keys=self.__sort_keys)
            json_repr = json_repr.replace('"{}"'.format(format_spec.format(id)), json_obj_repr)

        return json_repr

class ResizeImageANDCoord():
    def __init__(self, args):
        self.args = args
        self.ROOT = args.data_path
        
        self.PROJECT_NM = 'V01_DSME_Project 2022-08-03 14_54_01'
        if args.project == 'V01':
            self.PROJECT_NM = 'V01_DSME_Project 2022-08-03 14_54_01'
        elif args.project == 'V02':
            self.PROJECT_NM = 'V02_AKER_Project 2022-08-05 12_17_05'
        elif args.project == 'V03':
            self.PROJECT_NM = 'V03_KBR_Project 2022-08-03 14_56_42'
        elif args.project == 'V04':
            self.PROJECT_NM = 'V04_TECH_Project 2022-08-05 12_16_43'

        self.ORIGIN_PATH = os.path.join(self.ROOT, self.PROJECT_NM)
        self.SAVE_PATH = os.path.join(self.ROOT, '{}_resize'.format(self.PROJECT_NM))
        self.SAVE_JSON_PATH = os.path.join(self.SAVE_PATH, 'annotations')
        os.makedirs(self.SAVE_JSON_PATH, exist_ok=True)
        self.idx = 1



    def resize_img_save_coord(self, scale, re_height, re_width, img_dict, anno_dict):

        coord_tp = anno_dict["bbox"]
                    
        x1 = coord_tp[0] * scale
        y1 = coord_tp[1] * scale
        x2 = (coord_tp[0] + coord_tp[2]) * scale
        y2 = (coord_tp[1] + coord_tp[3]) * scale

        anno_dict['bbox'] = [x1, y1, x2-x1, y2-y1]
        anno_dict['area'] = (x2-x1) * (y2-y1)
        img_dict['height'], img_dict['width'] = re_height, re_width

        return img_dict, anno_dict, [x1, y1, x2, y2]

    def run_for_merge_dataset(self):
        for dataform in ['train', 'val', 'test']:
            # data\PnID\V01_DSME_Project 2022-08-03 14_54_01\annotations
            origin_json_filepath = os.path.join(self.ORIGIN_PATH, 'annotations', 'instances_{}.json'.format(dataform))
            save_img_dir = os.path.join(self.SAVE_PATH, dataform)
            os.makedirs(save_img_dir, exist_ok=True)

            images = []
            annotations = []

            with open(origin_json_filepath, 'r', encoding='utf-8') as jf:
                origin_json = json.load(jf)

            images_df = pd.DataFrame(origin_json['images'])
            annos_df = pd.DataFrame(origin_json['annotations'])
            print('len(images) = {}, len(annotations) = {}'.format(len(images_df), len(annos_df)))

            for img_dict in tqdm.tqdm(origin_json['images']):
                img_path = os.path.join(self.ORIGIN_PATH, dataform, img_dict["file_name"])

                try:
                    # Load image (PIL, convert np.array)
                    img = Image.open(img_path)
                    img = ImageOps.exif_transpose(img)  # EXIF 에 카메라 사진 촬영 각도가 반대로 들어가 있는 경우, 변환
                    
                    img_arr = np.asarray(img)

                    scale = self.args.img_size / img_arr.shape[0]
                    re_height = self.args.img_size
                    re_width = int(img_arr.shape[1] * scale)

                    # 이미지 저장
                    img_resize_lanczos = img.resize((re_width, re_height), Image.LANCZOS)
                    img_resize_lanczos.save(os.path.join(save_img_dir, os.path.basename(img_path)))

                    for anno_dict in annos_df[annos_df['image_id']==img_dict['id']].to_dict('records'):

                        new_img_dict, anno_dict, coord_list = self.resize_img_save_coord(scale, re_height, re_width, img_dict, anno_dict)

                        annotations.append(anno_dict)

                    images.append(new_img_dict)
                    
                except OSError as oe:
                    print(oe)
                    print('깨진 이미지 입니다. 해당 이미지는 학습 대상에서 제외됩니다.')
                    pass
                except ValueError as ve:
                    print(ve)

            print('len(images) = {}, len(annotations) = {}'.format(len(images), len(annotations)))
            
            origin_json['images'] = pd.DataFrame(images).sort_values(by=['id']).to_dict('records')
            origin_json['annotations'] = pd.DataFrame(annotations).sort_values(by=['id']).to_dict('records')

            with open(os.path.join(self.SAVE_JSON_PATH, "instances_{}.json".format(dataform)), "w", encoding='utf-8') as json_file:
                json_file.write(json.dumps(origin_json, cls=MyEncoder, indent=4, ensure_ascii=False))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default='V01',help="Config file's directory name OR Project name")
    parser.add_argument('--data_path', type=str, default='data/PnID',help="Config file's directory name OR Project name")
    parser.add_argument('--img_size', type=int, default=2048, help="resize value, default value for efficientDet3 is 896")

    args = parser.parse_args()
    riac=ResizeImageANDCoord(args)
    riac.run_for_merge_dataset()

