## 어노테이션 객체들의 정보로 통계를 내는 부분

| 파일 | 설명 |
|---|:---|
| stat_bbox_meta.py | 처음 받는 labels, meta로 나누어진 JSON 파일들을 읽어 객체들의 정보를 모아 CSV로 출력 |
| stat_from_instance_json.py | labels, meta를 종합하여 만든 instances JSON 파일에서 객체들의 정보를 모아 CSV로 출력 |
| csv_stat.py | CSV 파일을 읽어 최대, 최소, 평균 등 간단한 통계를 출력 => 전체의 통계를 내기보다는 값을 필터링하여 통계를 내보기 위함 |
