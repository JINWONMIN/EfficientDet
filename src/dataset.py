'''
author: Min-jinwon
Date: 2022-12-09
'''

import argparse, os

from utils.utils import Params
from utils.datasets.merge_instances import CreateLearningJson
from utils.datasets.split_dataset import SplitDataset
from utils.datasets.tiling import Tiling 


parser = argparse.ArgumentParser('Create Dataset - mjw')
parser.add_argument('--config', type=str, default='./config/202212/V02/V02.yml', required=True,
                    help='config file path')
args = parser.parse_args()

params = Params(args.config)

def main():
    print('start')
    print('merge raw dataset')
    create_json = CreateLearningJson(default_path=params.rawdata_path,
                                     save_path=params.ROOT,
                                     project_type=params.project_name,
                                     config_json=params.cfg_json,
                                     config_csv=params.cfg_csv)
    create_json.run()
    print('complete')
    
    print('start')
    print('split train / test / validation')
    split_dataset = SplitDataset(root=params.ROOT,
                                 img_path=os.path.join(params.rawdata_path, params.project_name),
                                 project_type=params.project_name,
                                 img_threshold=params.img_threshold,
                                 th_yn=params.th_yn,
                                 fold=params.fold)
    split_dataset.split()
    print('complete \n')
    
    print('start')
    print('tiling dataset \n')
    
    print('train')
    tiling_train = Tiling(root=params.ROOT,
                          cpu_core=params.cpu_core,
                          data_type=params.train_set,
                          project_type=params.project_name,
                          rows=params.rows,
                          cols=params.cols,
                          overlap_height_ratio=params.overlap_height_ratio,
                          overlap_width_ratio=params.overlap_width_ratio)
    tiling_train.main()
    tiling_train.merge()
    
    print('val')
    tiling_train = Tiling(root=params.ROOT,
                          cpu_core=params.cpu_core,
                          data_type=params.val_set,
                          project_type=params.project_name,
                          rows=params.rows,
                          cols=params.cols,
                          overlap_height_ratio=params.overlap_height_ratio,
                          overlap_width_ratio=params.overlap_width_ratio)
    tiling_train.main()
    tiling_train.merge()
    
    print('test')
    tiling_train = Tiling(root=params.ROOT,
                          cpu_core=params.cpu_core,
                          data_type=params.test_set,
                          project_type=params.project_name,
                          rows=params.rows,
                          cols=params.cols,
                          overlap_height_ratio=params.overlap_height_ratio,
                          overlap_width_ratio=params.overlap_width_ratio)
    tiling_train.main()
    tiling_train.merge()
    print('complete')

if __name__ == "__main__":
    main()
    
    