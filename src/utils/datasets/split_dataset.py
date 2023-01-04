'''
author: Min-jinwon
Date: 2022-12-09
'''

import json
from xmlrpc.client import boolean
import funcy
from sklearn.model_selection import train_test_split
import os
from _ctypes import PyObj_FromPtr
from utils.datasets.utils import filter_annotations, filter_categories, move_image_with_rootpath
from utils.datasets.utils import split_dataset, load_images, save_coco


class SplitDataset:
    def __init__(self, root: str, img_path: str, project_type: str, img_threshold: int, th_yn: bool, fold: int) -> None:
        
        self.data_path = root
        self.ori_img_path = img_path
        self.img_threshold = img_threshold
        self.th_yn = th_yn
        self.fold = fold
        self.project_type = project_type
    
        self.annotations_path = os.path.join(self.data_path, self.project_type, 'annotations')
        self.merged_instance_path = os.path.join(self.annotations_path, 'origin_instances_all.json')

    def split(self):
        with open(self.merged_instance_path, 'rt', encoding='UTF-8') as annotations:
            coco = json.load(annotations)
            images = coco['images']
            annotations = coco['annotations']
            categories = coco['categories']

            images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)
            images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

            train_anno, _anno = split_dataset(annotations, self.fold, 5)
            test_anno, val_anno = split_dataset(_anno, self.fold, 2)
            
            train_img = load_images(train_anno, self.merged_instance_path)
            test_img = load_images(test_anno, self.merged_instance_path)
            val_img = load_images(val_anno, self.merged_instance_path)
                        
            train_img = sorted(train_img, key=lambda train_img: (train_img['id']))
            test_img = sorted(test_img, key=lambda test_img: (test_img['id']))
            val_img = sorted(val_img, key=lambda val_img: (val_img['id']))
                    
            curr_anno = 0
            for idx in range(len(train_anno)):
                curr_anno += 1
                train_anno[idx]["id"] = curr_anno
            for idx in range(len(test_anno)):
                curr_anno += 1
                test_anno[idx]["id"] = curr_anno
            for idx in range(len(val_anno)):
                curr_anno += 1
                val_anno[idx]["id"] = curr_anno  
            
            # category id 초기화    
            cate = filter_categories(categories, train_anno)
            cate_name = sorted([int(x['name']) for x in cate])
            cate_dict = {n: i+1 for i, n in enumerate(cate_name)}
            
            for i in range(len(train_anno)):
                train_anno[i]["category_id"] = cate_dict.get(train_anno[i]['category_name'])
            for i in range(len(test_anno)):
                test_anno[i]["category_id"] = cate_dict.get(test_anno[i]['category_name'])
            for i in range(len(val_anno)):
                val_anno[i]["category_id"] = cate_dict.get(val_anno[i]['category_name'])
            
            save_path = os.path.join(self.data_path, self.project_type)
            save_coco(file=self.annotations_path, dataform="train", images=train_img, annotations=train_anno,
                        categories=cate)
            move_image_with_rootpath(image_dict=train_img, save_dir=save_path, img_root=self.ori_img_path,
                                        dataform="train", img_threshold=self.img_threshold, th_yn=self.th_yn)
            
            save_coco(file=self.annotations_path, dataform="test", images=test_img, annotations=test_anno,
                        categories=cate)
            move_image_with_rootpath(image_dict=test_img, save_dir=save_path, img_root=self.ori_img_path,
                                        dataform="test", img_threshold=self.img_threshold, th_yn=self.th_yn)
            
            save_coco(file=self.annotations_path, dataform="val", images=val_img, annotations=val_anno,
                        categories=cate)
            move_image_with_rootpath(image_dict=val_img, save_dir=save_path, img_root=self.ori_img_path,
                                        dataform="val", img_threshold=self.img_threshold, th_yn=self.th_yn)


            print("Saved train({}) entries in {} and test({}) in {} and val({}) in {}".format(len(train_img), self.annotations_path,
                                                                            len(test_img), self.annotations_path,
                                                                            len(val_img), self.annotations_path))
            