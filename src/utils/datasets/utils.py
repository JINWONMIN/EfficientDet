'''
author: Min-jinwon
Date: 2022-12-09
'''
import funcy
import os, glob, json
import cv2
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pycocotools.coco import COCO
from .merge_instances import MyEncoder
import warnings

warnings.filterwarnings('ignore')
 

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)

def filter_images(images, annotations):
    annotation_ids = funcy.lmap(lambda i: int(i['image_id']), annotations)
    return funcy.lfilter(lambda a: int(a['id']) in annotation_ids, images)

def filter_categories(categories, annotations):
    category_ids = funcy.lmap(lambda i: int(i['category_id']), annotations)
    category = funcy.lfilter(lambda a: int(a['id']) in category_ids, categories)
    
    category1 = sorted(category, key=lambda x: x['name'])
    for i in range(len(category1)):
        category1[i]['id'] = i+1
    return category1

def move_image_with_rootpath(image_dict, save_dir, dataform, img_root, img_threshold=210, th_yn=False):
    for idx in range(len(image_dict)):
        file_name = image_dict[idx]["file_name"]
        dict_imgname = file_name.rstrip('.png')

        origin_images = []
        for ext in ('jpg', 'png', 'jpeg'):
            origin_images.extend(glob.glob('{}/**/{}.{}'.format(img_root, dict_imgname, ext)))
        origin_path = origin_images[0]

        image_dict[idx]['file_name'] = '{}'.format(image_dict[idx]['file_name'])
        
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
            print(save_dir + ' created')
        if not os.path.isdir(os.path.join(save_dir, 'ori_split')):
            os.mkdir(os.path.join(save_dir, 'ori_split'))
            print(os.path.join(save_dir, 'ori_split') + ' created')
        if not os.path.isdir(os.path.join(save_dir, 'ori_split', dataform)):
            os.mkdir(os.path.join(save_dir, 'ori_split', dataform))
            print(os.path.join(save_dir, 'ori_split', dataform) + ' created')
            
        save_path = os.path.join(save_dir, 'ori_split', dataform, image_dict[idx]['file_name'])
        if not os.path.exists(save_path):
            if th_yn == True:
                load = cv2.imread(origin_path)
                img = cv2.cvtColor(load, cv2.COLOR_BGR2RGB)
                _, img1 = cv2.threshold(img, img_threshold, 255, cv2.THRESH_BINARY)
                cv2.imwrite(origin_path, img1)
                shutil.copy(origin_path, save_path)
            else:
                shutil.copy(origin_path, save_path)

    return image_dict

def filter_dataform_images(images, annotations):
    annotations = annotations.reshape(-1).tolist()
    images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)
    images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)
    return images
        
def split_dataset(annotation: list, fold=0, ratio=10):
    df = pd.DataFrame(annotation)
    df = df[df['category_id'] != 'unknown']
    
    skf = StratifiedKFold(n_splits=ratio, shuffle=True, random_state=44)
    df_folds = df[['image_id']].copy()
    
    df_folds.loc[:, 'bbox_count'] = 1
    df_folds = df_folds.groupby('image_id').count()
    df_folds.loc[:, 'object_count'] = df.groupby('image_id')['category_id'].nunique()
    
    df_folds.loc[:, 'stratify_group'] = np.char.add(
        df_folds['object_count'].values.astype(str),
        df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
    )
    
    df_folds.loc[:, 'fold'] = fold
    for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
        df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number
        
    df_folds.reset_index(inplace=True)
    df_valid = pd.merge(df, df_folds[df_folds['fold'] == fold], on='image_id')
    df_train = pd.merge(df, df_folds[df_folds['fold'] != fold], on='image_id')
    
    df_train.drop(columns=['bbox_count', 'object_count', 'stratify_group', 'fold'], axis=1, inplace=True)
    df_valid.drop(columns=['bbox_count', 'object_count', 'stratify_group', 'fold'], axis=1, inplace=True)
    
    train_list = df_train.to_dict('record')
    valid_list = df_valid.to_dict('record')
    
    return train_list, valid_list

def load_images(anno: list, json_path: str):
    '''
    json_path: original json file path
    '''
    coco_ = COCO(json_path)
    img_ids = [x['image_id'] for x in anno]
    img_ids = set(img_ids)
    
    img_list = []
    for img_id in img_ids:
        img_list.append(coco_.loadImgs(img_id))
    img_list = [x for i in img_list for x in i]
    
    return img_list

def save_coco(file, dataform, images, annotations, categories):
    with open(os.path.join(file, f'ori_instances_{dataform}.json'), 'wt', encoding='utf-8') as coco:
        coco.write(json.dumps({'images': images, 'annotations': annotations, 'categories': categories},
                              cls=MyEncoder, indent=4, ensure_ascii=False))