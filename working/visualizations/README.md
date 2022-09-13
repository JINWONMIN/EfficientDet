## 어노테이션을 그리는(=레이블링) 부분

1. JSON 파일에서 images와 annotations를 읽어온 후
2. 읽어온 객체들의 정보로 원본 사진들에 어노테이션

> image path: 원본 사진들이 위치한 폴더의 경로
>
> label_json path: 어노테이션 정보를 담은 JSON 파일의 경로
>
> save path: 어노테이션이 그려진 사진들이 저장될 (상위) 경로
> -> 이 안에서 train-val-test 3가지 폴더로 구분하여 사진 저장

---

> 65, 84 Line) 'V01_test' 부분으로 save path의 하위 폴더 이름 설정
>
> 74 Line) bbox는 어노테이션 좌표가 x Min, y Min, Width, Height를 의미 (x Min, y Min, x Max, y Max)를 의미하는 경우도 있으니 확인 필요)
> 
>
> 
>
> 
>
> 
