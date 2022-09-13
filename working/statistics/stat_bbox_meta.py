import os
import json
import csv
import matplotlib.pyplot as plt

DATA_PATH = r'\\192.168.219.150\XaiData\R&D\Project\2022_06_NIA_조선·해양 플랜트 P&ID 심볼 식별 데이터\02.데이터\62_SampleData_1cycle-2022-0805\V04_TECH_Project 2022-08-05 12_16_43'

class StatBBOX:
    def __init__(self, DATA_PATH: str):
        self.data_path = DATA_PATH
        self.total_stat = self.all_bbox()
        # self.scatter_graph()
        self.to_csv()

    def all_bbox(self):
        total_stat = {}
        id = 1
        with os.scandir(self.data_path + r'\meta\V04_05_057') as entries:
            for entry in entries:
                with open(entry, 'r', encoding='utf-8') as f:
                    meta_json = json.load(f)
                    f.close()
                img_name = meta_json['data_key']
                label_path = meta_json['label_path']
                print(len(label_path))
                with open(self.data_path + f'/{label_path[0]}', 'r') as j:
                    labe_json = json.load(j)
                    for a in labe_json['objects']:
                        total_stat[id] = {'image': img_name, 'class': a['class_name'], 'width': a['annotation']['coord']['width'], 'height': a['annotation']['coord']['height']}
                        id += 1
        return total_stat

    def scatter_graph(self, key: str):
        x_values = []
        y_values = []
        # for key in self.total_stat:
        for v in self.total_stat[key].values():
            x_values.append(v[0])
            y_values.append(v[1])
        plt.scatter(x_values, y_values, s=1.0)
        plt.grid()
        plt.title(key)
        plt.show()

    def to_csv(self):
        target_dict = self.total_stat
        with open('Sample_V04_statistics.csv', 'w', newline='') as csvfile:
            header_key = ['id', 'image', 'class', 'width', 'height']
            val = csv.DictWriter(csvfile, fieldnames=header_key)
            val.writeheader()
            for key in target_dict:
                val.writerow(
                    {'id': key, 'image': target_dict[key]['image'], 'class': target_dict[key]['class'], 'width': target_dict[key]['width'], 'height': target_dict[key]['height']})


if __name__ == '__main__':
    result = StatBBOX(DATA_PATH)