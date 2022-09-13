import os
import json
import csv
import matplotlib.pyplot as plt

INST_PATH = r'C:\Users\xaiplanet\Desktop\P&ID\data'

class StatBBOX:
    def __init__(self, JSON_PATH: str):
        self.json_path = JSON_PATH
        self.total_stat = self.all_bbox()
        # self.scatter_graph()  # 산점도 그리기
        self.to_csv()  # 어노테이션 id, width, height 3열의 csv 만들기
        
    def all_bbox(self):
        total_stat = {}
        json_filter = '.json'
        with os.scandir(self.json_path) as entries:
            for entry in [e for e in entries if e.name.endswith(json_filter)]:
                with open(entry, 'r') as f:
                    json_file = json.load(f)
                    f.close()
                total_stat[entry.name[:-5]] = {a['id']: {'class': a['class_name'], 'width': a['bbox'][2], 'height': a['bbox'][3]} for a in json_file['annotations']}
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
        for f in self.total_stat:
            target_dict = self.total_stat[f]
            with open(f'{f}_statistics.csv', 'w', newline='') as csvfile:
                header_key = ['id', 'class', 'width', 'height']
                val = csv.DictWriter(csvfile, fieldnames=header_key)
                val.writeheader()
                for key in target_dict:
                    val.writerow({'id': key, 'class': target_dict[key]['class'], 'width': target_dict[key]['width'], 'height': target_dict[key]['height']})

if __name__ == '__main__':
    StatBBOX(INST_PATH)