#!usr/bin/env python
import argparse
import os, json
import re
from collections import Counter
import tqdm
from PIL import Image
import pandas as pd

'''

create merged instance file for using _01_split_image.py

usage: python _01_split_image.py --rows {num_rows} --cols {num_cols} --reverse {merge_yn} --square {square_yn} 

'''


class SplitImage():
    def __init__(self):
        self.images_df = []
        self.annos_df = []
        self.new_image_id = 1
        self.new_img_list = []
        self.new_anno_id = 0
        self.new_anno_list = []


    def split(self, im, rows, cols, image_path, img_dict):
        save_img_dir = 'data/PnID/V01_DSME_Project 2022-08-03 14_54_01_divided'
        im_width, im_height = im.size
        row_width = int(im_width / rows)
        row_height = int(im_height / cols)
        
        n = 1
        for i in range(0, cols):
            for j in range(0, rows):
                new_img_dict = {}
                box = (j * row_width, i * row_height, j * row_width +
                    row_width, i * row_height + row_height) # (xmin, ymin, xmax, ymax)
                outp = im.crop(box)
                name, ext = os.path.splitext(os.path.basename(image_path))
                new_name = name + "_" + str(n) + ext
                outp_path = os.path.join(save_img_dir, new_name)
                image_yn = False

                # new annotation
                for anno_dict in self.annos_df[self.annos_df['image_id']==img_dict['id']].to_dict('records'):
                    new_anno_dict = {}
                    coord = anno_dict['bbox']
                    coord_tp = [coord[0], coord[1], coord[0]+coord[2], coord[1]+coord[3]] 
                    if coord_tp[0] >= box[0] and coord_tp[2] <= box[2]:
                        if coord_tp[1] >= box[1] and coord_tp[3] <= box[3]:
                            image_yn = True
                            self.new_anno_id += 1
                            new_anno_dict['id'] = self.new_anno_id
                            new_anno_dict['image_id'] = self.new_image_id
                            new_anno_dict['category_id'] = anno_dict['category_id']
                            new_anno_dict['category_name'] = anno_dict['category_name']
                            new_anno_dict['class_name'] = anno_dict['class_name']
                            new_anno_dict['bbox'] = [coord[0]-box[0], coord[1]-box[1], coord[2], coord[3]]
                            new_anno_dict['area'] = anno_dict['area']
                            new_anno_dict['iscrowd'] = 0
                            self.new_anno_list.append(new_anno_dict)
                    

                # new image
                if image_yn:
                    new_img_dict["id"] = self.new_image_id
                    new_img_dict["file_name"] = new_name
                    new_img_dict["width"] = outp.size[0]
                    new_img_dict["height"] = outp.size[1]

                    self.new_img_list.append(new_img_dict)
                    self.new_image_id += 1
                    print("Exporting image tile: " + outp_path)
                    outp.save(outp_path)
                n += 1



    def reverse_split(paths_to_merge, rows, cols, image_path):
        if len(paths_to_merge) == 0:
            print("No images to merge!")
            return
        for index, path in enumerate(paths_to_merge):
            path_number = int(path.split("_")[-1].split(".")[0])
            if path_number != index:
                print("Warning: Image " + path +
                    " has a number that does not match its index!")
                print("Please rename it first to match the rest of the images.")
                return
        images_to_merge = [Image.open(p) for p in paths_to_merge]
        image1 = images_to_merge[0]
        new_width = image1.size[0] * cols
        new_height = image1.size[1] * rows
        print(paths_to_merge)
        new_image = Image.new('RGB', (new_width, new_height), (250, 250, 250))
        print("Merging image tiles with the following layout:", end=" ")
        for i in range(0, rows):
            print("\n")
            for j in range(0, cols):
                print(paths_to_merge[i * cols + j], end=" ")
        print("\n")
        for i in range(0, rows):
            for j in range(0, cols):
                image = images_to_merge[i * cols + j]
                new_image.paste(image, (j * image.size[0], i * image.size[1]))
        print("Saving merged image: " + image_path)
        new_image.save(image_path)
        new_image.show()



    def determine_bg_color(self, im):
        print("Determining background color...")
        im_width, im_height = im.size
        rgb_im = im.convert('RGBA')
        all_colors = []
        areas = [[(0, 0), (im_width, im_height / 10)],
                [(0, 0), (im_width / 10, im_height)],
                [(im_width * 9 / 10, 0), (im_width, im_height)],
                [(0, im_height * 9 / 10), (im_width, im_height)]]
        for area in areas:
            start = area[0]
            end = area[1]
            for x in range(int(start[0]), int(end[0])):
                for y in range(int(start[1]), int(end[1])):
                    pix = rgb_im.getpixel((x, y))
                    all_colors.append(pix)
        return Counter(all_colors).most_common(1)[0][0]


    def main(self, args):

        merged_json_filepath = './data/PnID/V01_DSME_Project 2022-08-03 14_54_01/annotations'

        # save_img_dir = 'data/PnID/V01_DSME_Project 2022-08-03 14_54_01_divided'
        # os.makedirs(save_img_dir, exist_ok=True)

        if args.reverse:
            print(
                "Reverse mode selected! Will try to merge multiple tiles of an image into one.")
            print("\n")
            start_name, ext = os.path.splitext(image_path)
            # Find all files that start with the same name as the image,
            # followed by "_" and a number, and with the same file extension.
            expr = re.compile(r"^" + start_name + "_\d+" + ext + "$")
            paths_to_merge = sorted([f for f in os.listdir(
                os.getcwd()) if re.match(expr, f)])
            self.reverse_split(paths_to_merge, args.rows,
                        args.cols, image_path)
        else:
            with open(os.path.join(merged_json_filepath, "instances_all.json"), 'r', encoding='utf-8') as jf:
                json_str = json.load(jf)

            self.images_df = pd.DataFrame(json_str['images'])
            self.annos_df = pd.DataFrame(json_str['annotations'])
            new_json = {}

            for img_dict in tqdm.tqdm(json_str['images']):
                image_path = './raw_data/V01_DSME_Project 2022-08-03 14_54_01/images/V01_03_017/{}'.format(img_dict["file_name"])

                # image_path = args.image_path[0]
                if args.load_large_images:
                    Image.MAX_IMAGE_PIXELS = None

                im = Image.open(image_path)
                im_width, im_height = im.size
                min_dimension = min(im_width, im_height)
                max_dimension = max(im_width, im_height)
                if args.square:
                    print("Resizing image to a square...")
                    bg_color = self.determine_bg_color(im)
                    print("Background color is... " + str(bg_color))
                    im_r = Image.new("RGBA", (max_dimension, max_dimension), bg_color)
                    offset = int((max_dimension - min_dimension) / 2)
                    if im_width > im_height:
                        im_r.paste(im, (0, offset))
                    else:
                        im_r.paste(im, (offset, 0))
                    self.split(im_r, args.rows, args.cols, image_path)
                    print("Exporting resized image...")
                    im_r.save(image_path + "_squared.png")
                else:
                    # HERE
                    self.split(im, args.rows, args.cols, image_path, img_dict) # (xmin, ymin, xmax, ymax)

            new_json["images"] = self.new_img_list
            new_json["annotations"] = self.new_anno_list
            new_json["categories"] = json_str['categories']

            with open(os.path.join(merged_json_filepath, "instances_all_divide.json"), "w", encoding='utf-8') as json_file:
                json_file.write(json.dumps(new_json, indent=4, ensure_ascii=False))

            print("Total image len = {}".format(len(self.new_img_list)))
            print("Total annotation len = {}".format(len(self.new_anno_list)))
            print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split an image into rows and columns.")
    parser.add_argument("--rows", type=int, default=4, nargs='?',
                        help="How many rows to split the image into (horizontal split).")
    parser.add_argument("--cols", type=int, default=4, nargs='?',
                        help="How many columns to split the image into (vertical split).")
    parser.add_argument("-r", "--reverse", action="store_true",
                    help="Reverse the splitting process, i.e. merge multiple tiles of an image into one.")
    parser.add_argument("-s", "--square", action="store_true",
                        help="If the image should be resized into a square before splitting.")
    # parser.add_argument("--cleanup", action="store_true",
    #                     help="After splitting or merging, delete the original image/images.")
    # parser.add_argument("--load-large-images", action="store_true",
    #                     help="Ignore the PIL decompression bomb protection and load all large files.")
    args = parser.parse_args()
    si = SplitImage()
    si.main(args)