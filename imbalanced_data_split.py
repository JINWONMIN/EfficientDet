import json
import argparse
import funcy
from sklearn.model_selection import train_test_split
# from skmultilearn.model_selection import iterative_train_test_split
import numpy as np
import glob
import shutil
import os
import re
from _ctypes import PyObj_FromPtr


'''
create instances_all.json file each project by using the gen_dataset.py before using this script. 

usage: python imbalanced_data_split.py --having-annotations --multi-class


152번 라인에 50개 초과 anno info 제거 코드는 아직 미완성 이라 주석 처리 했습니다.
프로젝트 별로 parser.argv default 경로 변경 하면 됩니다.
추후에 경로 들어가는거 수정 해서 최대한 변경 안 되게끔 하겠습니다.
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


def save_coco(file, dataform, images, annotations, categories):
    os.makedirs(file, exist_ok=True)
    with open(os.path.join(file, f'instances_{dataform}.json'), 'wt', encoding='utf-8') as coco:
        coco.write(json.dumps({'images': images, 'annotations': annotations, 'categories': categories},
                              cls=MyEncoder, indent=4, ensure_ascii=False))


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def filter_images(images, annotations):
    annotation_ids = funcy.lmap(lambda i: int(i['image_id']), annotations)
    return funcy.lfilter(lambda a: int(a['id']) in annotation_ids, images)


def filter_categories(categories, annotations):
    category_ids = funcy.lmap(lambda i: int(i['category_id']), annotations)
    return funcy.lfilter(lambda a: int(a['id']) in category_ids, categories)


def convert_id_annot(annotations):
    for idx in range(len(annotations)):
        if annotations[idx]['category_name'] == "07-15":
            annotations[idx]['category_id'] = 1
        elif annotations[idx]['category_name'] == "07-16":
            annotations[idx]['category_id'] = 2
        elif annotations[idx]['category_name'] == "07-20":
            annotations[idx]['category_id'] = 3
        elif annotations[idx]['category_name'] == "07-26":
            annotations[idx]['category_id'] = 4
        elif annotations[idx]['category_name'] == "09-07":
            annotations[idx]['category_id'] = 5
        else:
            annotations[idx]['category_id'] = 6
    return annotations


def convert_id_cate(categories):
    for idx in range(len(categories)):
        if categories[idx]['name'] == "07-15":
            categories[idx]['id'] = 1
        elif categories[idx]['name'] == "07-16":
            categories[idx]['id'] = 2
        elif categories[idx]['name'] == "07-20":
            categories[idx]['id'] = 3
        elif categories[idx]['name'] == "07-26":
            categories[idx]['id'] = 4
        elif categories[idx]['name'] == "09-07":
            categories[idx]['id'] = 5
        else:
            categories[idx]['id'] = 6
    return categories

def __move_image_with_rootpath(image_dict, save_dir, dataform, img_root, prj_type):
    for idx in range(len(image_dict)):
        file_name = image_dict[idx]["file_name"]
        dict_imgname = file_name.rstrip('.png')

        origin_images = []
        for ext in ('jpg', 'png', 'jpeg'):
            origin_images.extend(glob.glob('{}/{}/images/**/{}.{}'.format(img_root, prj_type, dict_imgname, ext)))
        origin_path = origin_images[0]

        image_dict[idx]['file_name'] = '{}'.format(image_dict[idx]['file_name'])
        os.makedirs(os.path.join(save_dir, dataform), exist_ok=True)
        save_path = os.path.join(save_dir, dataform, image_dict[idx]['file_name'])
        if not os.path.exists(save_path):
            shutil.copy(origin_path, save_path)

    return image_dict


def filter_dataform_images(images, annotations):
    annotations = annotations.reshape(-1).tolist()
    images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)
    images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)
    return images


parser = argparse.ArgumentParser(description='Splits annotations file into training and test, val sets.')
parser.add_argument('--annotations', metavar='annotations', type=str, required=False,
                    default="./data/PnID_divide/V01_DSME_Project 2022-08-03 14_54_01/labels/instances_all_divide.json",
                    help='Path to annotations file.')
parser.add_argument('--train', type=str, help='Where to store training annotations',
                    default=r"./data/PnID/V01_DSME_Project 2022-08-03 14_54_01_divided/annotations", required=False)
parser.add_argument('--test', type=str, help='Where to store test annotations',
                    default=r"./data/PnID/V01_DSME_Project 2022-08-03 14_54_01_divided/annotations", required=False)
parser.add_argument('--val', type=str, help='Where to store val annotations',
                    default=r"./data/PnID/V01_DSME_Project 2022-08-03 14_54_01_divided/annotations", required=False)
parser.add_argument('--s', dest='split', type=float, required=False, default=0.8,
                    help="A percentage of a split; a number in (0, 1)")
parser.add_argument('--having-annotations', dest='having_annotations', action='store_true',
                    help='Ignore all images without annotations. Keep only these with at least one annotation')

parser.add_argument('--multi-class', dest='multi_class', action='store_true',
                    help='Split a multi-class dataset while preserving class distributions in train and test, val sets')

args = parser.parse_args()


def main(args):
    save_dir = r"../data/PnID/V01_DSME_Project 2022-08-03 14_54_01_divided"
    img_root = r"./data/PnID_divide"
    prj_type = r"V01_DSME_Project 2022-08-03 14_54_01"

    with open(args.annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        number_of_images = len(images)

        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

        if args.having_annotations:
            images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        if args.multi_class:

            annotation_categories = funcy.lmap(lambda a: int(a['category_id']), annotations)
            annotation_categories = funcy.lremove(lambda i: annotation_categories.count(i) <= 90, annotation_categories)
            # annotation_categories = funcy.lwithout(lambda i: )

            annotations = funcy.lremove(lambda i: i['category_id'] not in annotation_categories, annotations)
            images = filter_images(images, annotations)

            X_train, X_vt = train_test_split(images,
                                             test_size=1 - args.split,
                                            )

            X_test, X_valid = train_test_split(X_vt,
                                               test_size=0.5,
                                               )

            for idx in range(len(X_train)):
                X_train = sorted(X_train, key=lambda X_train: (X_train['id']))
            for idx in range(len(X_test)):
                X_test = sorted(X_test, key=lambda X_test: (X_test['id']))
            for idx in range(len(X_valid)):
                X_valid = sorted(X_valid, key=lambda X_valid: (X_valid['id']))

            curr_anno = 0
            anno_train = filter_annotations(annotations, X_train)
            for idx in range(len(anno_train)):
                curr_anno += 1
                anno_train[idx]["id"] = curr_anno
            anno_test = filter_annotations(annotations, X_test)
            for idx in range(len(anno_test)):
                curr_anno += 1
                anno_test[idx]["id"] = curr_anno
            anno_val = filter_annotations(annotations, X_valid)
            for idx in range(len(anno_val)):
                curr_anno += 1
                anno_val[idx]["id"] = curr_anno

            cate_train = filter_categories(categories, anno_train)
            cate_test = filter_categories(categories, anno_test)
            cate_val = filter_categories(categories, anno_val)

            save_coco(file=args.train, dataform="train", images=X_train, annotations=convert_id_annot(anno_train),
                      categories=convert_id_cate(cate_train))
            __move_image_with_rootpath(image_dict=X_train, save_dir=save_dir, img_root=img_root,
                                       dataform="train", prj_type=prj_type)
            save_coco(file=args.test, dataform="test", images=X_test, annotations=convert_id_annot(anno_test),
                      categories=convert_id_cate(cate_test))
            __move_image_with_rootpath(image_dict=X_test, save_dir=save_dir, img_root=img_root,
                                       dataform="test", prj_type=prj_type)
            save_coco(file=args.val, dataform="val", images=X_valid, annotations=convert_id_annot(anno_val),
                      categories=convert_id_cate(cate_val))
            __move_image_with_rootpath(image_dict=X_valid, save_dir=save_dir, img_root=img_root,
                                       dataform="val", prj_type=prj_type)

            # X_train, X_vt, y_train, y_vt = train_test_split(np.array([annotations]).T,
            #                                                 np.array([annotation_categories]).T,
            #                                                 test_size=1 - args.split,
            #                                                 random_state=42,
            #                                                 shuffle=True,
            #                                                 stratify=np.array([annotation_categories]).T)
            #
            # X_valid, X_test, y_valid, y_test = train_test_split(X_vt,
            #                                                 y_vt,
            #                                                 test_size=0.5,
            #                                                 random_state=42,
            #                                                 shuffle=True,
            #                                                 stratify=y_vt)

            # save_coco(file=args.train, dataform="train", images=filter_images(images, X_train.reshape(-1)),
            #           annotations=X_train.reshape(-1).tolist(), categories=categories)
            # __move_image_with_rootpath(image_dict=filter_dataform_images(images, X_train), save_dir=save_dir,
            #                            img_root=img_root, dataform="train", prj_type=prj_type)
            #
            # save_coco(file=args.test, dataform="test", images=filter_images(images, X_test.reshape(-1)),
            #           annotations=X_test.reshape(-1).tolist(), categories=categories)
            # __move_image_with_rootpath(image_dict=filter_dataform_images(images, X_test), save_dir=save_dir,
            #                            img_root=img_root, dataform="test", prj_type=prj_type)
            #
            # save_coco(file=args.val, dataform="val", images=filter_images(images, X_valid.reshape(-1)),
            #           annotations=X_valid.reshape(-1).tolist(), categories=categories)
            # __move_image_with_rootpath(image_dict=filter_dataform_images(images, X_valid), save_dir=save_dir,
            #                            img_root=img_root, dataform="val", prj_type=prj_type)

            print("Saved {} entries in {} and {} in {} and {} in {}".format(len(X_train), args.train,
                                                                            len(X_test), args.test,
                                                                            len(X_valid), args.val))

        else:

            X_train, X_test = train_test_split(images, train_size=args.split)

            anno_train = filter_annotations(annotations, X_train)
            anno_test = filter_annotations(annotations, X_test)

            save_coco(file=args.train, dataform="train", images=X_train, annotations=anno_train, categories=categories)
            save_coco(file=args.test, dataform="test", images=X_test, annotations=anno_test, categories=categories)

            print("Saved {} entries in {} and {} in {}".format(len(anno_train), args.train, len(anno_test), args.test))


if __name__ == "__main__":
    main(args)