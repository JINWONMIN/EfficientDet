import os
import shutil
from glob import glob
import funcy


def match_image(IMAGE_PATH: str, INFERENCE_PATH: str) -> None:
    img_list = os.listdir(IMAGE_PATH)
    infer_list = os.listdir(INFERENCE_PATH)

    target_list = funcy.lfilter(lambda a: a in infer_list, img_list)
    for img in target_list:
        shutil.copy(IMAGE_PATH + '/' + img, INFERENCE_PATH + '/' + img + '.jpg')


def main(image_path: str, inference_path: str) -> None:
    match_image(image_path, inference_path)


if __name__ == "__main__":
    IMAGE_PATH = "../labeled_data/V01_divided_220901"
    INFERENCE_PATH = "../labeled_data/inference"
    main(IMAGE_PATH, INFERENCE_PATH)
