## EfficientDet Pytorch [Nia - P&ID]
---

Google Brain에서 2019년에 발표한 EfficientDet: Scalable and Efficient Object Detection 논문을 토대로 만든 zylo117의 깃허브를 참고.

- [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070.pdf)
- [zylo117: Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)

<br/>
<br/>

## EfficientDet Architectrue
---
![architecture](https://user-images.githubusercontent.com/94345086/210199690-4ca071e8-2e27-4656-a70c-a95564c34ca2.png)
![scaling configs](https://user-images.githubusercontent.com/94345086/210199692-0af87997-aad3-427b-aaf6-0f70887ebfef.png)

- compound coefficient: 4를 사용.
- Backbone은 Freeze 시키고 나머지 BiFPN과 HEAD 부분만 finetuning

<br/>
<br/>

## W/B download
---
```bash
$ mkdir weights

$ wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d0.pth -P ./weights
$ wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d1.pth -P ./weights
$ wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d2.pth -P ./weights
$ wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d3.pth -P ./weights
$ wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d4.pth -P ./weights
$ wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d5.pth -P ./weights
$ wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d6.pth -P ./weights
$ wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d7.pth -P ./weights
```

<br/>
<br/>

## Dataset
---
1. raw 데이터셋 준비
```bash
# your dataset structure should be like this
datasets/
    -your_project_name/
        -train_set_name/
            -*.jpg
        -val_set_name/
            -*.jpg
        -annotations
            -instances_{train_set_name}.json
            -instances_{val_set_name}.json

# for example, coco2017
datasets/
    -coco2017/
        -train2017/
            -000000000001.jpg
            -000000000002.jpg
            -000000000003.jpg
        -val2017/
            -000000000004.jpg
            -000000000005.jpg
            -000000000006.jpg
        -annotations
            -instances_train2017.json
            -instances_val2017.json
```
2. manual set script's specific parameters
```yml
ROOT: './data/PnID_updated_202212'  # the root folder of dataset

# Use dataset.py
rawdata_path: './raw_data/PnID_updated_202212'
cfg_json: './config/202212/V01/V01.json'
cfg_csv: './config/202212/V01/categories_V01.csv'

fold: 0
th_yn: False
img_threshold: 210

cpu_core: 60
rows: 5
cols: 5
overlap_height_ratio: 0.2
overlap_width_ratio: 0.2
```
> python dataset.py --config {config file/path/to}
```bash
$ python dataset.py --config config/V01/V01.yml
```

<br/>
<br/>

## Train
---
1. manual set script's specific parameters
```yaml
project_name: V01  # 데이터셋 생성시 _data/pid/ 하위에 생성되어있는 폴더명 기재
train_set: train
val_set: val
test_set: test

visual_path: origin

num_gpus: 4

mean: [ 0.485, 0.456, 0.406 ]
std: [ 0.229, 0.224, 0.225 ]

anchors_scales: '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'
anchors_ratios: '[(0.7, 1.4), (1.0, 1.0), (1.4, 0.7)]'  # train / inference

input_sizes: [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

obj_list: ["1102", "1201", "1202", "1206", "1209", "1210", "1211", "1212", "1301", "1401", "1501", "1502", "1504", "1505",
"1506", "1507", "1508", "1509", "1511", "1512", "1513", "1514", "1517", "1518", "1519", "1523", "1524", "1525", "1526", "1528",
"1530", "1533", "1535", "1603", "1701", "1704", "1706", "1707", "1709", "1710", "1711", "1713", "1715", "1716", "1717", "1719",
"1722", "1723", "1725", "1726", "1733", "1734", "1735", "1736", "1742", "1743", "1744", "1746", "1747", "1749", "1751", "1752",
"1754", "1755", "1757", "1758", "1759", "1801", "1810", "1812", "1813", "1903", "1907", "1908", "1909", "1913", "1920", "1921",
"1922", "1927", "1933", "1934", "1935", "1937", "1938", "1942", "1946", "1947", "1951", "1952", "1954", "1955", "1956", "1958",
"1962", "1963", "1965", "1966", "1967", "1968", "1969", "2001", "2102", "2103"]

ROOT: './data/PnID_updated_202212'  # the root folder of dataset

# Use train.py / eval.py / fo_eval.py
compound_coef: 4

# Use train.py
num_workers: 32 # num_workers of dataloader
batch_size: 24  # The number of images per batch among all devices
lr: 0.001 
optim: 'adamw'  # select optimizer for training,
                # suggest using 'admaw' until the very final stage then switch to 'sgd'
num_epochs: 100
val_interval: 1 # Number of epoches between valing phases
save_interval: 10000 # Number of steps between saving
es_min_delta: 0.0   # Early stopping's parameter: minimum change loss to qualify as an improvement
es_patience: 0  # Early stopping's parameter: number of epochs with no improvement after which training
                # will be stopped. Set to 0 to disable this technique
log_path: 'result/'
load_weights: 'weights/efficientdet-d4.pth' # whether to load weights from a checkpoint, set None to initialize,
                                            # set 'last' to load last checkpoint
saved_path: 'result/'
matched_threshold: 0.5  # Threshold for positive matches
unmatched_threshold: 0.4 # Threshold for negative matches
```
> python train.py --help
```bash
$ python train.py --config config/V01/V01.yml --head_only True --debug False
```

<br/>
<br/>

## Test
---
1. manual set script's specific parameters
```yaml
# Use eval.py
weights: 'result/V01/20221220_221332/efficientdet-d4_15_47168_best.pth' # '/path/to/weights'
result_path: 'result/V01/20221220_221332/inference' # displing images 
nms_threshold: 0.5  # nms threshold, don't change it if not for testing purposes
device: 3 # eval.py / fo_eval.py
threshold: 0.2  # confidence score threshold
```
> python evaluate.py --help
```bash
$ python evaluate.py --config config/V01/V01.yml --cuda True --float16 False --override True --evaluate --display-bboxes
```

<br/>
<br/>

## FiftyOne evaluate
---
1. manual set script's specific parameters
```yml
### Use only fo_eval.py
obj_list1: ["None", "1102", "1201", "1202", "1206", "1209", "1210", "1211", "1212", "1301", "1401", "1501", "1502", "1504", "1505",
"1506", "1507", "1508", "1509", "1511", "1512", "1513", "1514", "1517", "1518", "1519", "1523", "1524", "1525", "1526", "1528",
"1530", "1533", "1535", "1603", "1701", "1704", "1706", "1707", "1709", "1710", "1711", "1713", "1715", "1716", "1717", "1719",
"1722", "1723", "1725", "1726", "1733", "1734", "1735", "1736", "1742", "1743", "1744", "1746", "1747", "1749", "1751", "1752",
"1754", "1755", "1757", "1758", "1759", "1801", "1810", "1812", "1813", "1903", "1907", "1908", "1909", "1913", "1920", "1921",
"1922", "1927", "1933", "1934", "1935", "1937", "1938", "1942", "1946", "1947", "1951", "1952", "1954", "1955", "1956", "1958",
"1962", "1963", "1965", "1966", "1967", "1968", "1969", "2001", "2102", "2103"]
model_name: '20221220_212149'
MAX_IMAGES: 100000
iou_threshs: [0.5]
iou: 0.5
```
> python evaluate.py --help
```bash
$ python fo_eval.py --config config/V01/V01.yml --cuda True --float16 False --override True 
```

<br/>
<br/>

---
###Open Sourc LISENCE   
- Voxel51 - Binary forms redistribution
    - it was used as a binary form in the fo_eval.py
<https://github.com/voxel51/fiftyone/blob/develop/LICENSE>
