import os
import json
import csv
import matplotlib.pyplot as plt

LABE_PATH = r'\\192.168.219.150\XaiData\R&D\Project\2022_06_NIA_조선·해양 플랜트 P&ID 심볼 식별 데이터\02.데이터\62_SampleData_1cycle-2022-0805\V01_DSME_Project 2022-08-03 14_54_01\labels'


class StatBBOX:
    def __init__(self, JSON_PATH: str):
        self.json_path = JSON_PATH
        self.total_stat = self.all_bbox()
        # self.scatter_graph()
        self.to_csv()

    def all_bbox(self):
        total_stat = {}
        id = 1
        with os.scandir(self.json_path) as entries:
            for entry in entries:
                with open(entry, 'r') as f:
                    json_file = json.load(f)
                    f.close()
                for a in json_file['objects']:
                    total_stat[id] = {'width': a['annotation']['coord']['width'], 'height': a['annotation']['coord']['height']}
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
        labels = ['id', 'width', 'height']
        # for key in self.total_stat:
        target_dict = self.total_stat
        with open('V01_sample_statistics.csv', 'w', newline='') as csvfile:
            header_key = ['id', 'width', 'height']
            val = csv.DictWriter(csvfile, fieldnames=header_key)
            val.writeheader()
            for key in target_dict:
                val.writerow(
                    {'id': key, 'width': target_dict[key]['width'], 'height': target_dict[key]['height']})


if __name__ == '__main__':
    result = StatBBOX(LABE_PATH)