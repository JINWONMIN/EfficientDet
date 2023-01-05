## 구동 환경 세팅
---
![](https://user-images.githubusercontent.com/94345086/210198204-19891af5-b0e6-4915-863a-69ab4d5e99ef.png)

docker file과 scripts를 같은 경로에 위치.


<br/>

- docker image build
```bash
$ docker build -f dev.dockerfile -t {user name}/{image name}:{tag} .

$ docker build -f dev.dockerfile -t mjw/efficientdet:v.0.1 .
```
- docker container run
```bash
$ docker run --runtime=nvidia --rm -itd --gpus all --name eff-det-01 mjw/efficientdet:v0.1 .

$ docker exec -it eff-det-01 /bin/bash

*** In Docker **
root@dd78a3ae73ec:/# $ cd /home
```
