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

Execute _01_merge_instance.py first.
You can split dataset into three type(train, valid, test).

usage: python _02_gen_dataset_imbalanced.py --data_path {} --ori_img_path {} --having-annotations --multi-class

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
        if annotations[idx]['category_id'] == 50:
            annotations[idx]['category_id'] = 1
        elif annotations[idx]['category_id'] == 51:
            annotations[idx]['category_id'] = 2
        elif annotations[idx]['category_id'] == 55:
            annotations[idx]['category_id'] = 3
        elif annotations[idx]['category_id'] == 61:
            annotations[idx]['category_id'] = 4
        elif annotations[idx]['category_id'] == 98:
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
        save_path = os.path.join(save_dir, dataform, image_dict[idx]['file_name'])
        if not os.path.exists(save_path):
            shutil.copy(origin_path, save_path)

    return image_dict


def filter_dataform_images(images, annotations):
    annotations = annotations.reshape(-1).tolist()
    images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)
    images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)
    return images


def main(args):
    annotations_path = os.path.join(args.data_path, 'annotations')
    merged_instance_path = os.path.join(annotations_path, 'instances_all.json')
    prj_type = os.path.basename(os.path.normpath(args.data_path))

    with open(merged_instance_path, 'rt', encoding='UTF-8') as annotations:
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
            annotation_categories = funcy.lremove(lambda i: annotation_categories.count(i) <= 100, annotation_categories)

            annotations = funcy.lremove(lambda i: i['category_id'] not in annotation_categories, annotations)
            images = filter_images(images, annotations)

            X_train, X_vt = train_test_split(images,
                                             test_size=1 - args.split,
                                             random_state=42,
                                             )
            X_test, X_valid = train_test_split(X_vt,
                                               test_size=0.5,
                                               random_state=42)

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

            save_coco(file=annotations_path, dataform="train", images=X_train, annotations=convert_id_annot(anno_train),
                      categories=convert_id_cate(cate_train))
            __move_image_with_rootpath(image_dict=X_train, save_dir=args.data_path, img_root=args.ori_img_path,
                                       dataform="train", prj_type=prj_type)
            save_coco(file=annotations_path, dataform="test", images=X_test, annotations=convert_id_annot(anno_test),
                      categories=convert_id_cate(cate_test))
            __move_image_with_rootpath(image_dict=X_test, save_dir=args.data_path, img_root=args.ori_img_path,
                                       dataform="test", prj_type=prj_type)
            save_coco(file=annotations_path, dataform="val", images=X_valid, annotations=convert_id_annot(anno_val),
                      categories=convert_id_cate(cate_val))
            __move_image_with_rootpath(image_dict=X_valid, save_dir=args.data_path, img_root=args.ori_img_path,
                                       dataform="val", prj_type=prj_type)


            print("Saved {} entries in {} and {} in {} and {} in {}".format(len(X_train), annotations_path,
                                                                            len(X_test), annotations_path,
                                                                            len(X_valid), annotations_path))

        else:

            X_train, X_test = train_test_split(images, train_size=args.split)

            anno_train = filter_annotations(annotations, X_train)
            anno_test = filter_annotations(annotations, X_test)

            save_coco(file=annotations_path, dataform="train", images=X_train, annotations=anno_train, categories=categories)
            save_coco(file=annotations_path, dataform="test", images=X_test, annotations=anno_test, categories=categories)

            print("Saved {} entries in {} and {} in {}".format(len(anno_train), annotations_path, len(anno_test), annotations_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Splits annotations file into training and test, val sets.')
    parser.add_argument('--data_path', type=str, help='Where to store training annotations',
                        default=r"./data/PnID/V01_DSME_Project 2022-08-03 14_54_01", required=False)
    parser.add_argument('--ori_img_path', type=str, help='original image path',
                        default=r"./raw_data/SampleData", required=False)
    parser.add_argument('--s', dest='split', type=float, required=False, default=0.8,
                        help="A percentage of a split; a number in (0, 1)")
    parser.add_argument('--having-annotations', dest='having_annotations', action='store_true',
                        help='Ignore all images without annotations. Keep only these with at least one annotation')
    parser.add_argument('--multi-class', dest='multi_class', action='store_true',
                        help='Split a multi-class dataset while preserving class distributions in train and test, val sets')

    args = parser.parse_args()
    main(args)