# P&ID
#### project 하위 디렉토리 설명

| 폴더명 | 설   명 |
| ---------------- | ---------------- |
| config           | 설정관련 파일 저장                    |
| data             | 데이터 관련 파일 저장                 |
| result           | 수행결과 저장                         |
| weights | 기본 모델 저장  |
| _01_merge_instance.py | 데이터셋 병합 |
| _02_gen_dataset_imbalanced.py | 데이터셋 생성 (_01_merge_instance.py 선 실행 필) |
| _03_resize_image.py | 이미지 사이즈 변경 |
| _04_gen_anchor_ratios.py | 학습 데이터에 따른 anchor ratio 출력 |
| _05_train.py | 모델 학습 수행                         			|
| _06_evaluate.py | 모델 평가 수행                    |
| _06_infer_test.py | 모델 추론 수행 |
| _07_matching_inference.pyt | match pred image to ground images and save|
| requirements.txt 		   | 실행환경                         			|



## 환경 설치

- 환경설치

  1. windows OS인 경우, visual studio 2015 버전 이상 설치되어있어야 함 (linux는 해당되지 않음)

     - pycocotools 패키지의 의존패키지

     - 윈도우즈용 버전 설치

       https://daewonyoon.tistory.com/327

       [https://bblib.net/entry/파이토치윈도우에서-pycocotools-사용하는-방법](https://bblib.net/entry/파이토치윈도우에서-pycocotools-사용하는-방법)

       ```jsx
       pip install cython
       conda install git
       
       pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
       pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
       ```

  2. conda 가상환경 생성

     ```
     conda create -n nia_pnid python=3.7
     activate nia_pnid
     ```

     

  3. 필요 라이브러리 설치

     - torch 공식 홈페이지에서 로컬 환경에 맞게 설치 명령어를 받아서 설치할 것

       https://pytorch.org/get-started/locally/

       ```python
       pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
       
       pip install nbconvert
       pip install pycocotools 
       pip install webcolors 
       pip install pyyaml tqdm opencv-python
       pip install tensorboard
       pip install tensorboardX
       ```

       



## 소스코드 활용

### 1) 데이터 병합

> 1. config/categories.csv 정의
>     ex) supercategory,id,name
>             V01,1,07-15
>             V01,2,07-16
>
>     cf) supercategory는 각 프로젝트 구분에 따름 (project name 과 무관함)
>             V01_DSME_Project 2022-08-03 14_54_01 = V01
>             V02_AKER_Project 2022-08-05 12_17_05 = V02
>     		V03_KBR_Project 2022-08-03 14_56_42   = V03
>             V04_TECH_Project 2022-08-05 12_16_43 = V04
>
>     
>
> 2. 해당 클래스가 있는 원본 폴더 확인
>     NAS: r"\\192.168.219.150\XaiData\R&D\Project\2022_06_NIA_조선·해양 플랜트 P&ID 심볼 식별 데이터\02.데이터\62_SampleData_1cycle-2022-0805" 하위
>     ex) 62_SampleData_1cycle-2022-0805
>             ㄴ V01_DSME_Project 2022-08-03 14_54_01
>                ㄴ images
>                ㄴ labels
>                ㄴ meta
>                ㄴ project.json
>
>     
>
> 3. 실행
>     python _01_merge_instance.py --ctg_path {catagory_path}
>
>     - catagory_path은 categories.csv 경로를 입력
>       - 만약 폴더명에 띄어쓰기가 존재하는 경우에는 소스코드 내부 최하단의 argument parser의 default 값을 변경하는 것을 권장함
>     - 소스코드 내부 CreateLearningJson 클래스 __init__() 내 PATH 수정 필요

```bash
python _01_merge_instance.py --ctg_path {catagory_path}
```



### 2) 데이터셋 생성

> 1. 병합된 instances_all.json 파일이 존재하는 지 확인
> 2. 실행
>    python _02_gen_dataset_imbalanced.py  --data_path {data_path} --ori_img_path {ori_img_path} --having-annotations --multi-class
>    - 만약 폴더명에 띄어쓰기가 존재하는 경우에는 소스코드 내부 최하단의 argument parser의 default 값을 변경하는 것을 권장함

```bash
python _02_gen_dataset_imbalanced.py --data_path {data_path} --ori_img_path {ori_img_path} --having-annotations --multi-class
```



### 3) 이미지 resize

> 1. 실행
>    python _03_resize_image.py  --data_path {data_path} --ori_img_path {ori_img_path} --having-annotations --multi-class
>    - 만약 폴더명에 띄어쓰기가 존재하는 경우에는 소스코드 내부 최하단의 argument parser의 default 값을 변경하는 것을 권장함

```bash
python _03_resize_image.py --project {project_name} --data_path {data_path} --img_size {image_size}
```



### 4) config 파일 설정

> 모델 config 파일 설정
>
> config/V01.yml

```
project_name: V01_DSME_Project 2022-08-03 14_54_01  # 데이터셋 생성시 data/PnID/ 하위에 생성되어있는 폴더명 기재
train_set: train
val_set: val
test_set: test
num_gpus: 1

mean: [ 0.485, 0.456, 0.406 ]
std: [ 0.229, 0.224, 0.225 ]

anchors_scales: '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'
anchors_ratios: '[(0.7, 1.4), (1.0, 1.0), (1.4, 0.7)]' 	# python _04_gen_anchor_ratios.py 의 결과 반영

# obj_list == class_list
# 데이터셋 생성시 사용한 categories.csv 와 동일한 순서로 아래 list에 클래스 정의
obj_list: ['07-15','07-16','07-20','07-26','09-07','09-08']
```

```
python _04_gen_anchor_ratios.py --project {project_name} --data_path {data_path} --compound_coef {compound_coef}

# 실행결과를 yml파일의 anchors_ratios 에 반영
optimal_anchors_ratios=[(0.9, 1.1), (1.3, 0.8), (1.7, 0.6)]

```



### 5) EfficientDet Model Training

> 모델 학습 수행
```
python _05_train.py
```
> 실행방법 : 아래의 파라미터 수정 후 위의 명령어 실행
>  --project : config의 yml파일명(default : 'V01')
>  --compound_coef : EfficientDet 버전 (default : 3)
>  --batch_size : 배치 사이즈 (default : 1)
>  --num_epochs: 에포크 수 (default : 100)
>  --data_path : 학습 데이터셋 경로 (default : 'data/PnID')
>  --log_path : 로그 저장 경로 (default : 'result/')
>  --load_weights : 모델 경로 (default : 'weights/efficientdet-d3.pth')
>  --saved_path : 저장 경로 (default : 'result/')

이 외 나머지 옵션은 default 그대로 사용할 것

> 실행결과 : result 폴더 내 모델 pth 파일 생성



### 6) EfficientDet Model Evaluate 
> 테스트 데이터셋에 대한 모델 평가 수행 
```
python _06_evaluate.py --project {} --compound_coef {} --weights {} --data_path {} 
```
> 실행방법 : 아래의 파라미터 수정 후 위의 명령어 실행
>  --project : config의 yml파일명(default : V01')
>  --compound_coef : EfficientDet 버전 (default : 3)
>  --weights : 학습 모델 경로 (default : 'weights/efficientdet-d3.pth')
>--data_path :  데이터셋 경로 (default : 'data/PnID')
> 
>실행결과 : 데스트 데이터셋에 대한 검증결과 출력 및 result 하위에 좌표 정보 json 파일 저장




### 7) EfficientDet Model Inference
> 다건의 이미지에 대해 모델 추론을 수행
```
python _06_infer_test.py --project {} --compound_coef {} --weights {} --data_path {}
```
> 실행방법 : 아래의 파라미터 수정 후 위의 명령어 실행
>  --project : config의 yml파일명 (default : 'V01')
>  --compound_coef : EfficientDet 버전 (default : 3)
>  --weights : 학습 모델 경로 (default : 'weight/efficientdet-d3.pth')
>  --data_path : 데이터셋 경로 (default : 'data/PnID')

> 실행결과 : result/{project}/{now}.jpg  추론된 이미지 저장

> 추론된 이미지를 정답 이미지랑 매칭 후 정답 이미지 추론된 이미지 경로에 저장
'''
python _07_matching_inference.py
'''
> 실행방법 : 코드의 IMAGE_PATH 와 INFERENCE_PATH 만 경로에 맞춰서 수정
