import csv
import pandas as pd
import os

PATH = './count_220901'
FILE = r'\\192.168.219.150\XaiData\R&D\Project\2022_06_NIA_조선·해양 플랜트 P&ID 심볼 식별 데이터\02.데이터\statistics\62_SampleData_1cycle-2022-0805\Sample_V04_statistics.csv'

# with os.scandir(PATH) as entries:
#     # for folder in [e for e in entries if e.is_dir()]:
#     # with os.scandir(folder) as entries:
#     for entry in [e for e in entries if e.is_file() and e.name.endswith('.csv')]:
csv_df = pd.read_csv(FILE)
csv_df = csv_df[(csv_df['width'] <= 30) & (csv_df['height'] <= 30)]
statistics = csv_df[['width', 'height']].describe()
# statistics = csv_df['class'].value_counts()
# statistics.to_csv(r'./{0}_describe.csv'.format(entry.name[:-15]), sep=',')
statistics.to_csv(r'./Sample_V04_statistics_describe.csv', sep=',')


                

print()