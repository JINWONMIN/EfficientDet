'''
author: Min-jinwon
Date: 2022-12-09
'''

import os, json, glob
from PIL import Image
import pandas as pd

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000

import multiprocessing
from multiprocessing import Pool
from datetime import datetime
from pycocotools.coco import COCO
 

class Tiling:
    def __init__(self, root: str, cpu_core: int, data_type: str, project_type: str, 
                 rows: int, cols: int, overlap_height_ratio: float=0.2, overlap_width_ratio: float=0.2) -> None:
        self.root = root
        self.cpu_core = cpu_core
        self.data_type = data_type
        self.project_type = project_type
        self.rows = rows
        self.cols = cols
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        
        self.images_df = []
        self.annos_df = []
        self.new_image_id = 1
        self.new_img_list = []
        self.new_anno_id = 0
        self.new_anno_list = []
        self.slice_bboxes = []

    def split(self, img_dict):
        rows = self.rows
        cols = self.cols
        project_type = self.project_type
        data_type = self.data_type
        
        image_path = '{}/{}/ori_split/{}/{}'.format(self.root, self.project_type ,self.data_type ,img_dict["file_name"])

        save_img_dir = f'{self.root}/{project_type}/{data_type}'
        os.makedirs(save_img_dir, exist_ok=True)

        im = Image.open(image_path)
        im_width, im_height = im.size
        slice_width = int(im_width / rows)
        slice_height = int(im_height / cols)
        
        """
        Given the height and width of an image, calculates how to divide the image into
        overlapping slices according to the height and width provided. These slices are returned
        as bounding boxes in xyxy format
        """
        check_list = []
        n = 0
        y_max = y_min = 0
        y_overlap = int(self.overlap_height_ratio * slice_height)
        x_overlap = int(self.overlap_width_ratio * slice_width)
        while y_max < im_height:
            x_min = x_max = 0
            y_max = y_min + slice_height
            while x_max < im_width:
                x_max = x_min + slice_width
                
                n += 1
                if y_max > im_height or x_max > im_width:
                    xmax = min(im_width, x_max)
                    ymax = min(im_height, y_max)
                    xmin = max(0, xmax - slice_width)
                    ymin = max(0, ymax - slice_height)
                    slice_bbox = [xmin, ymin, xmax, ymax]
                    # print('____slice_bbox:', slice_bbox)
                    
                    new_img_dict = {}
                    outp = im.crop(slice_bbox)
                    name, ext = os.path.splitext(os.path.basename(image_path))
                    new_name = name + "_0" + str(n) + ext
                    outp_path = os.path.join(save_img_dir, new_name)
                    image_yn = False # annotations 정보 없는 이미지도 저장
                    
                    # new annotation
                    for anno_dict in self.annos_df[self.annos_df['image_id'] == img_dict["id"]].to_dict("records"):
                        new_anno_dict = {}
                        coord = anno_dict['bbox']
                        coord_tp = [coord[0], coord[1], coord[0] + coord[2], coord[1] + coord[3]]
                        if coord_tp[0] >= slice_bbox[0] and coord_tp[2] <= slice_bbox[2]:
                            if coord_tp[1] >= slice_bbox[1] and coord_tp[3] <= slice_bbox[3]:
                                if data_type == 'test':
                                    image_yn = True   # image_yn이 False로 설정 되어있을 때 사용
                                    self.new_anno_id += 1
                                    new_anno_dict["id"] = self.new_anno_id
                                    new_anno_dict['image_id'] = self.new_image_id
                                    new_anno_dict['category_id'] = anno_dict['category_id']
                                    new_anno_dict['category_name'] = anno_dict['category_name']
                                    new_anno_dict['class_name'] = anno_dict['class_name']
                                    new_anno_dict['bbox'] = [round(coord[0]-slice_bbox[0], 1), round(coord[1]-slice_bbox[1], 1) , coord[2] , coord[3]]
                                    new_anno_dict['area'] = anno_dict['area']
                                    new_anno_dict['iscrowd'] = 0
                                
                                    check_list.append([coord[0], coord[1], coord[2], coord[3]])
                                    self.new_anno_list.append(new_anno_dict)
                                else:
                                    if [coord[0], coord[1], coord[2], coord[3]] in check_list:
                                        pass
                                    else:
                                        image_yn = True   # image_yn이 False로 설정 되어있을 때 사용
                                        self.new_anno_id += 1
                                        new_anno_dict["id"] = self.new_anno_id
                                        new_anno_dict['image_id'] = self.new_image_id
                                        new_anno_dict['category_id'] = anno_dict['category_id']
                                        new_anno_dict['category_name'] = anno_dict['category_name']
                                        new_anno_dict['class_name'] = anno_dict['class_name']
                                        new_anno_dict['bbox'] = [round(coord[0]-slice_bbox[0], 1), round(coord[1]-slice_bbox[1], 1) , coord[2] , coord[3]]
                                        new_anno_dict['area'] = anno_dict['area']
                                        new_anno_dict['iscrowd'] = 0
                                    
                                        check_list.append([coord[0], coord[1], coord[2], coord[3]])
                                        self.new_anno_list.append(new_anno_dict)
    
                    # new image
                    if image_yn:
                        new_img_dict["id"] = self.new_image_id
                        new_img_dict["file_name"] = new_name
                        new_img_dict["width"] = outp.size[0]
                        new_img_dict["height"] = outp.size[1]
                        new_img_dict["slices"] = slice_bbox
                        new_img_dict["ori_width"] = im_width
                        new_img_dict["ori_height"] = im_height
                        new_img_dict["ori_file_name"] = os.path.basename(image_path)
                        
                        self.new_img_list.append(new_img_dict)
                        self.new_image_id += 1
                        print("Exporting image tile: " + outp_path)
                        outp.save(outp_path)
                    
                else:
                    slice_bbox = [x_min, y_min, x_max, y_max]
                    # print('slice_bbox:', slice_bbox)
                    new_img_dict = {}
                    outp = im.crop(slice_bbox)
                    name, ext = os.path.splitext(os.path.basename(image_path))
                    new_name = name + "_0" + str(n) + ext
                    outp_path = os.path.join(save_img_dir, new_name)
                    image_yn = False # annotations 정보 없는 이미지도 저장
                    
                    # new annotation
                    for anno_dict in self.annos_df[self.annos_df['image_id'] == img_dict["id"]].to_dict("records"):
                        new_anno_dict = {}
                        coord = anno_dict['bbox']
                        coord_tp = [coord[0], coord[1], coord[0] + coord[2], coord[1] + coord[3]]
                        if coord_tp[0] >= slice_bbox[0] and coord_tp[2] <= slice_bbox[2]:
                            if coord_tp[1] >= slice_bbox[1] and coord_tp[3] <= slice_bbox[3]:
                                if data_type == 'test':
                                    image_yn = True   # image_yn이 False로 설정 되어있을 때 사용
                                    self.new_anno_id += 1
                                    new_anno_dict["id"] = self.new_anno_id
                                    new_anno_dict['image_id'] = self.new_image_id
                                    new_anno_dict['category_id'] = anno_dict['category_id']
                                    new_anno_dict['category_name'] = anno_dict['category_name']
                                    new_anno_dict['class_name'] = anno_dict['class_name']
                                    new_anno_dict['bbox'] = [round(coord[0]-slice_bbox[0], 1), round(coord[1]-slice_bbox[1], 1) , coord[2] , coord[3]]
                                    new_anno_dict['area'] = anno_dict['area']
                                    new_anno_dict['iscrowd'] = 0
                                
                                    check_list.append([coord[0], coord[1], coord[2], coord[3]])
                                    self.new_anno_list.append(new_anno_dict)
                                else:
                                    if [coord[0], coord[1], coord[2], coord[3]] in check_list:
                                        pass
                                    else:
                                        image_yn = True
                                        self.new_anno_id += 1
                                        new_anno_dict["id"] = self.new_anno_id
                                        new_anno_dict['image_id'] = self.new_image_id
                                        new_anno_dict['category_id'] = anno_dict['category_id']
                                        new_anno_dict['category_name'] = anno_dict['category_name']
                                        new_anno_dict['class_name'] = anno_dict['class_name']
                                        new_anno_dict['bbox'] = [round(coord[0]-slice_bbox[0], 1), round(coord[1]-slice_bbox[1], 1) , coord[2] , coord[3]]
                                        new_anno_dict['area'] = anno_dict['area']
                                        new_anno_dict['iscrowd'] = 0
                                        
                                        check_list.append([coord[0], coord[1], coord[2], coord[3]])
                                        self.new_anno_list.append(new_anno_dict)
                        
                    # new image
                    if image_yn:
                        new_img_dict["id"] = self.new_image_id
                        new_img_dict["file_name"] = new_name
                        new_img_dict["width"] = outp.size[0]
                        new_img_dict["height"] = outp.size[1]
                        new_img_dict["slices"] = slice_bbox
                        new_img_dict["ori_width"] = im_width
                        new_img_dict["ori_height"] = im_height
                        new_img_dict["ori_file_name"] = os.path.basename(image_path)
                        
                        self.new_img_list.append(new_img_dict)
                        self.new_image_id += 1
                        print("Exporting image tile: " + outp_path)
                        outp.save(outp_path)
                x_min = x_max - x_overlap
            y_min = y_max - y_overlap
        
        new_json = {}
        new_json['images'] = self.new_img_list
        new_json['annotations'] = self.new_anno_list
        merged_json_filepath = f'{self.root}/{self.project_type}/annotations/temp/{data_type}'
        os.makedirs(merged_json_filepath, exist_ok=True)
        today = datetime.today()
        if len(self.new_img_list) == 0:
            pass
        with open(os.path.join(merged_json_filepath, f"{today.microsecond}.json"), "w", encoding='utf-8') as json_file:
            json_file.write(json.dumps(new_json, indent=4, ensure_ascii=False))
            
    def main(self):
        merged_json_filepath = f'{self.root}/{self.project_type}/annotations'
        # multiprocessing shareed variable 을 사용할 수 있는 객체 (자료구조를 사용해서 생성할 수 있음.)
        manager = multiprocessing.Manager()
        dic = manager.dict()

        with open(os.path.join(merged_json_filepath, f"ori_instances_{self.data_type}.json"), 'r', encoding='utf-8') as jf:
            json_str = json.load(jf)

        self.images_df = pd.DataFrame(json_str['images'])
        self.annos_df = pd.DataFrame(json_str['annotations'])

        try:
            pool = Pool(processes=self.cpu_core)
            pool.map(self.split, json_str['images'], chunksize=1)
        finally:
            pool.close()
            pool.join()
        
    def merge(self):
        merged_json_filepath = f'{self.root}/{self.project_type}/annotations'
        with open(os.path.join(merged_json_filepath, f'ori_instances_{self.data_type}.json'), 'r', encoding='utf-8') as ori:
            cat = json.load(ori)
        
        img_list = list()
        anno_list = list()
        tiled_jlist = glob.glob(f'{self.root}/{self.project_type}/annotations/temp/{self.data_type}/*.json')
        
        img_id = 1
        anno_id = 1
        
        for file_path in tiled_jlist:
            coco_gt = COCO(file_path)
            with open(file_path, 'r', encoding='utf-8') as file:
                coco = json.load(file)
                
            for img in coco['images']:
                anno_ids = coco_gt.getAnnIds(img['id'])
                img_dict = {'id': img_id, 'file_name': img['file_name'], 'width': img['width'], 'height': img['height'],
                        'slices': img['slices'], 'ori_width': img['ori_width'], 'ori_height': img['ori_height'], 'ori_file_name': img['ori_file_name']}
                img_list.append(img_dict)
                for anno in anno_ids:
                    anno_dict = coco_gt.loadAnns(anno)
                    anno_dict[0]['id'] = anno_id
                    anno_dict[0]['image_id'] = img_id
                    anno_list.append(anno_dict)
                    anno_id += 1
                img_id += 1
        anno_list = [x for i in anno_list for x in i]
        new_json = dict()
        new_json['images'] = img_list
        new_json['annotations'] = anno_list
        new_json['categories'] = cat['categories']
        
        with open(os.path.join(merged_json_filepath, f"instances_{self.data_type}.json"), "w", encoding="utf-8") as json_file:
            json_file.write(json.dumps(new_json, indent=4, ensure_ascii=False))
            
        print("Total {} image len = {}".format(self.data_type, len(img_list)))
        print("Total {} annotation len = {}".format(self.data_type, len(anno_list)))
        print("Done!")
        