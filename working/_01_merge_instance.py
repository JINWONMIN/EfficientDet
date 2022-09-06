import os, sys, re, json, argparse, math
from collections import OrderedDict
from _ctypes import PyObj_FromPtr
import pandas as pd
import random
import glob
from tqdm import tqdm


'''

create merged instance file for using _01_gen_dataset_imbalanced.py

usage: python _01_merge_instance.py --ctg_path {catagory_path}

'''


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
        self.DEFAULT_PATH = r'raw_data'
        self.SAVE_PATH = r'data/PnID'
        self.image_id = 0
        self.kor_type = r'V01_DSME_Project 2022-08-03 14_54_01'
        self.limit = 51

    def __len__(self):
        return len(self.label_dict)

    def __setClassCd(self):
        # conf.json : 클래스명 - 코드 맵핑
        with open('./config/conf.json', 'r', encoding='utf-8') as f:
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
        class_name = ''.join([self.__setClassCd().get(_str, 'unknown')])
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


    # instances_all.json 저장
    def __save_json(self, result_dict):
        try:
            json_save_dir = os.path.join(self.SAVE_PATH, self.kor_type, "annotations")
            os.makedirs(json_save_dir, exist_ok=True)

            with open(os.path.join(json_save_dir, f"instances_all.json"), "w",
                      encoding='utf-8') as json_file:
                json_file.write(json.dumps(result_dict, cls=MyEncoder, indent=4, ensure_ascii=False))
        except Exception as e:
            raise e


    # json 형태 구성
    def __make_dataset(self, data_list):
        image_list = []
        anno_list = []

        curr_cnt = 0
        anno_cnt = 0

        self.class_dict = self.__setClassCd()

        for json_idx, jsonfile in enumerate(data_list):
            try:
                curr_cnt += 1
                with open(jsonfile, 'r', encoding='utf-8-sig') as meta_json_file:
                    meta_json_str = json.load(meta_json_file)
                    label_id = meta_json_str['label_id']

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
            self.__save_json(result_dict)


    # 타겟 데이터프레임 구성
    def __get_target_data(self, meta_file_list):
        df = pd.DataFrame(columns=['filepath'])
        random.shuffle(meta_file_list)
        for o_file in tqdm(meta_file_list):
            df = df.append({'filepath': o_file}, ignore_index=True)

        return df



    # Main
    def run(self, args):
        self.__setClass(args.ctg_path)

        root_path = '{}/{}/meta/**/*.json'.format(self.DEFAULT_PATH, self.kor_type)
        root_path = os.path.join(self.DEFAULT_PATH, self.kor_type, 'meta/**/*.json')
        print('root_path={}'.format(root_path))

        file_list = glob.glob(root_path, recursive=True)
        print('len(json_list)={}'.format(len(file_list)))

        data_df = self.__get_target_data(file_list)
        print('len(data_df)={}'.format(len(data_df)))

        json_list = data_df['filepath'].to_list()

        self.__make_dataset(json_list)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create instance_all.json for EfficientDet")
    parser.add_argument("--ctg_path", required=False, type=str, default="./config/categories_V01.csv")

    args = parser.parse_args()

    cj = CreateLearningJson()
    cj.run(args)
