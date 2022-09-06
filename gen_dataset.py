import os, sys, re, json, argparse, math
from collections import OrderedDict
from _ctypes import PyObj_FromPtr
import pandas as pd
import random
import shutil
import glob
from tqdm import tqdm
import numpy as np

from sklearn.model_selection import train_test_split


class NoIndent(object):
    # Value wrapper.
    def __init__(self, value):
        self.value = value


class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

    def __init__(self, **kwargs):
        # Save copy of any keyword argument values needed for use here.
        self.__sort_keys = kwargs.get('sort_keys', None)
        super(MyEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj))
                if isinstance(obj, NoIndent) else super(MyEncoder, self).default(obj))

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.
        json_repr = super(MyEncoder, self).encode(obj)  # Default JSON.
        # Replace any marked-up object ids in the JSON repr with the value returned
        # from the json.jumps() of the corresponding wrapped Python object.

        for match in self.regex.finditer(json_repr):
            # see https://codeutlity.org/a/15012814/355230
            _id = int(match.group(1))
            no_indent = PyObj_FromPtr(_id)
            json_obj_repr = json.dumps(no_indent.value, sort_keys=self.__sort_keys)
            # Replace the matched id string with json formatted representation
            # of the corresponding Python object.
            json_repr = json_repr.replace('"{}"'.format(format_spec.format(_id)), json_obj_repr)

        return json_repr


class CreateLearningJson:
    def __init__(self):
        super().__init__()

        self.class_dict = dict()
        self.label_dict = dict()
        self.meta_dict = dict()
        self.LIST_CLASS = list()
        self.DF_CLASS = list()
        self.DEFAULT_PATH = r'/SampleData'
        self.IMAGE_PATH = r'images/**'
        self.SAVE_PATH = r'../data/PnID'
        self.image_id = 0
        self.kor_type = r'V01_DSME_Project 2022-08-03 14_54_01'
        self.limit = 51

    def __len__(self):
        return len(self.label_dict)

    def __set_class(self):
        with open('../visualization/conf.json', 'r', encoding='utf-8') as f:
            class_dict =json.load(f)
        return class_dict

    def __setClass(self, filepath):
        df = pd.read_csv(filepath)
        df['name'] = df['name'].astype(str)
        df['id'] = df["id"].apply(lambda x: int(x) if x == x else "")
        self.DF_CLASS = df
        ts_trp = df.transpose()
        self.LIST_CLASS = list(ts_trp.to_dict().values())

    def __class_mapping(self, _str):
        class_name = ''.join([self.class_dict.get(_str, 'unknown')])
        return class_name

    def __printProgress(self, iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
        format_str = "{0:." + str(decimals) + "f}"
        percent = format_str.format(100 * (iteration / float(total)))
        filled_length = int(math.ceil(bar_length * iteration / float(total)))
        bar = '>' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write('\r%s |%s| %s %s %s%s %s' % (prefix, bar, str(iteration) + " / " + str(total), "|", percent, '%', suffix))

        if iteration == total:
            sys.stdout.write('\n')

        sys.stdout.flush()

    def __move_image_with_rootpath(self, image_dict, save_dir, root_path):
        file_name = image_dict["file_name"]
        dict_imgname = file_name.rstrip('.png')

        origin_images=[]
        for ext in ('jpg', 'png', 'jpeg'):
            origin_images.extend(glob.glob('{}/{}.{}'.format(root_path, dict_imgname, ext)))
        origin_path=origin_images[0]

        image_dict['file_name'] = '{}'.format(image_dict['file_name'])
        save_path = os.path.join(save_dir, image_dict['file_name'])
        if not os.path.exists(save_path):
            shutil.copy(origin_path, save_path)

        return image_dict


    # def __save_json(self, result_dict, dataform):
    def __save_json(self, result_dict):
        try:
            json_save_dir = os.path.join(self.SAVE_PATH, self.kor_type, "annotations")
            os.makedirs(json_save_dir, exist_ok=True)

            # with open(os.path.join(json_save_dir, f"instances_{dataform}.json"), "w",
            with open(os.path.join(json_save_dir, f"instances_all.json"), "w",
                      encoding='utf-8') as json_file:
                json_file.write(json.dumps(result_dict, cls=MyEncoder, indent=4, ensure_ascii=False))
        except Exception as e:
            raise e

    def __make_dataset(self, data_list, dataform='train'):
        image_list = []
        anno_list = []

        curr_cnt = 0
        anno_cnt = 0
        info_dict = OrderedDict()

        self.class_dict = self.__set_class()

        save_dir = os.path.join(self.SAVE_PATH, self.kor_type, dataform)
        os.makedirs(save_dir, exist_ok=True)

        for json_idx, jsonfile in enumerate(data_list):
            try:
                curr_cnt += 1
                with open(jsonfile, 'r', encoding='utf-8-sig') as meta_json_file:
                    meta_json_str = json.load(meta_json_file)
                    label_id = meta_json_str['label_id']

                temp = list()

                # if len(info_dict.keys()) < 1:
                #     # info
                #     info_dict["name"] = meta_json_str["info"]["name"]
                #     info_dict["description"] = meta_json_str["info"]["description"]

                # meta.json info
                self.image_id += 1
                image_dict = OrderedDict()
                image_dict["file_name"] = meta_json_str['data_key']
                image_dict["id"] = self.image_id
                image_dict["height"] = meta_json_str["image_info"]["height"]
                image_dict["width"] = meta_json_str["image_info"]["width"]

                # annotations
                with open(os.path.join(self.DEFAULT_PATH, self.kor_type, f'labels/{label_id}.json'),
                          'r', encoding='utf-8-sig') as label_json:
                    label_json_str = json.load(label_json)
                    self.label_dict = label_json_str

                    for idx in range(len(self.label_dict["objects"])):
                        annotation = label_json_str["objects"][idx]['annotation']
                        if len(annotation) == 0:
                            self.image_id -= 1
                            continue

                        else:

                            anno_cnt += 1
                            class_name = label_json_str['objects'][idx]['class_name']
                            bbox = annotation["coord"]

                            anno_dict = OrderedDict()

                            anno_dict["area"] = bbox['width'] * bbox['height']

                            anno_dict["class_name"] = class_name
                            anno_dict["category_name"] = self.__class_mapping(class_name)  # if not in mapping_tabel
                                                                                            # 'unknown' 리턴
                            target_list = [x for x in self.LIST_CLASS if anno_dict["category_name"] in x["name"]]

                            if len(target_list) == 1:
                                target_category = target_list[0]
                                anno_dict["category_id"] = target_category["id"]
                            else:
                                anno_dict["category_id"] = "NAN"

                            if bbox['width'] <= float(30) or bbox['height'] <= float(30):
                                continue
                            else:
                                anno_dict["bbox"] = [bbox['x'], bbox['y'], bbox['width'], bbox['height']]
                                anno_dict["iscrowd"] = 0

                                anno_dict["id"] = anno_cnt
                                anno_dict["image_id"] = self.image_id

                                anno_list.append(anno_dict)

                                # if self.image_id == image_dict["id"]:
                                #     image_dict = self.__move_image_with_rootpath(image_dict, save_dir,
                                #                                                  root_path=os.path.join(self.DEFAULT_PATH,
                                #                                                                         self.kor_type,
                                #                                                                         self.IMAGE_PATH))
                    if image_dict['id'] == anno_dict["image_id"]:
                        image_list.append(image_dict)

                self.__printProgress(curr_cnt, len(data_list), 'Progress:', 'Complete', 1, 50)

            except FileNotFoundError as fe:
                print(fe)
                pass
            except json.decoder.JSONDecodeError as je:
                print(je)
                print(jsonfile)
                pass

            result_dict = {"images": image_list, "annotations": anno_list, "categories": self.LIST_CLASS}
            # self.__save_json(result_dict, dataform)
            self.__save_json(result_dict)

    def __get_target_data(self, meta_file_list):
        df = pd.DataFrame(columns=['filepath'])
        random.shuffle(meta_file_list)
        for o_file in tqdm(meta_file_list):
            df = df.append({'filepath': o_file}, ignore_index=True)
            # fault_code = '-'.join(os.path.basename(o_file).split('_')[1:3])
            # target_row = self.DF_CLASS[self.DF_CLASS['name'] == fault_code]
            # 데이터분포 불균형으로 undersampling 적용
            # if len(target_row) == 10 and len(df[df['fault_code'] == fault_code]) < self.limit:
            #     df = df.append({'fault_code': fault_code, 'filepath': o_file}, ignore_index=True)
        # df = df.sort_values('fault_code')
        return df

    # def __get_target_data(self, meta_file_list):
    #     self.class_dict = self.__set_class()
    #     df = pd.DataFrame(columns=['filepath', 'class_id'])
    #     random.shuffle(meta_file_list)
    #     for o_file in tqdm(meta_file_list):
    #         with open(o_file, 'r', encoding='utf-8') as meta_json:
    #             meta = json.load(meta_json)
    #             label_id = meta["label_id"]
    #         with open(os.path.join(self.DEFAULT_PATH, self.kor_type, f'labels/{label_id}.json'), 'r',
    #                   encoding='utf-8') as label_json:
    #             label = json.load(label_json)
    #             for idx in range(len(label["objects"])):
    #                 fault_code = self.__class_mapping(label['objects'][idx]['class_name'])
    #                 df = df.append({'filepath': o_file, 'class_id': fault_code}, ignore_index=True)
        # df.set_index("filepath", drop=False, inplace=True)
        # df = df.groupby('filepath')
        #
        # pd.set_option('display.max_row', 500)
        # pd.set_option('display.max_columns', 100)
        # print(df)
        # return df


    # Main
    def run(self, args):
        self.__setClass(args.ctg_path)

        root_path = '{}/{}/meta/**/*.json'.format(self.DEFAULT_PATH, self.kor_type)

        root_path = os.path.join(self.DEFAULT_PATH, self.kor_type, 'meta/**/*.json')
        # if args.part_name is not None:
        #     root_path = os.path.join(self.DEFAULT_PATH, self.kor_type, args.part_name, 'meta/V01_03_017/*.json')
        print('root_path={}'.format(root_path))

        file_list = glob.glob(root_path, recursive=True)
        print('len(json_list)={}'.format(len(file_list)))

        data_df = self.__get_target_data(file_list)
        print('len(data_df)={}'.format(len(data_df)))

        json_list = data_df['filepath'].to_list()
        # json_list_label = data_df['fault_code'].to_list()

        # X_train, X_vt = train_test_split(json_list, test_size=args.testset_rt, random_state=42, shuffle=True)
        # X_valid, X_test = train_test_split(X_vt, test_size=0.5, random_state=42, shuffle=True)

        self.__make_dataset(json_list, dataform='train')
        # self.__make_dataset(X_train, dataform='train')
        # self.__make_dataset(X_valid, dataform='val')
        # self.__make_dataset(X_test, dataform='test')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create dataset for EfficientDet")
    parser.add_argument("--testset_rt", required=False, type=float, help="Test Set Ratio", default=0.2)
    parser.add_argument("--ctg_path", required=False, type=str, default="./config/categories_V01.csv")
    # parser.add_argument("--project", required=False, type=str, help="project name, yaml 파일과 동일한 명칭 이어야 함", default="ship")
    # parser.add_argument("--part_name", required=False, type=str, default=None)

    args = parser.parse_args()

    cj = CreateLearningJson()
    cj.run(args)
