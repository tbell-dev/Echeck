# 한전 E-check REST API

## 시스템 호환 정보

전자식 세그먼트 숫자 탐지, 인식 모델의 경우 GPU 자원을 사용하므로 사전에 그래픽카드 드라이버와 CUDA, CUDNN이 설치되어야 한다.

```bash
# os : ubuntu 18.04
# GPU : RTX-3090 x 1
Python==3.7.10
CUDA==11.2
CUDNN==11.2
tensorflow==2.5.0
```

소스코드를 내려받는다.

```bash
git clone https://github.com/tbell-dev/Echeck.git
```

## 프로젝트 구조

Tree view : 

```
Echeck
├── api.py
├── config.ini
├── darknet
│   └── libdarknet.so
├── darknet.py
├── image
├── model
│   ├── num_classifier
│   └── num_detector
├── requirements.txt
├── tfServing.py
└── yolo-cfg
    ├── emeter-yolov4-416.cfg
    ├── emeter-yolov4-416.data
    ├── emeter-yolov4-416.names
    ├── emeter-yolov4-608.cfg
    ├── emeter-yolov4-608.data
    └── emeter-yolov4-608.names
```

- `api.py` : Flask API를 실행하기 위한 main 함수들 코드
- `config.ini` : 모델 설정 파일 정보와 TF-Serving 주소, 수신 이미지 저장 경로 등 서버 설정 정보들을 담고 있다.
    - Detector : 숫자 탐지 모델의 설정파일 경로, weight 파일 경로를 지정해준다.
    - Classifier : 숫자 분류 모델의 가중치 데이터 경로, threshold를 설정해준다.
    - TF-Serving : tf-serving 모델의 저장 경로를 지정해준다.
    - API_Save : API로 수신하는 이미지를 저장할 경로를 지정해준다.
    - Flask : Flask API 서버의 호스트, 포트 명을 지정해준다.
- `darknet` : libdarknet.so 파일을 저장할 디렉토리이다.
- `darknet.py` : darknet weight 파일을 읽고 처리하는 파이썬 스크립트이다.
- `model` : 숫자 탐지(num_detector), 인식(num_classifier) 모델을 저장하는 디렉토리이다.
- `requirements.txt` : 필요 python 라이브러리 정보이다.
- `tfServing.py` : TF-Serving을 처리하는 모듈이다.
- `yolo-cfg` : 숫자 탐지 모델의 설정파일들이다.

프로젝트 루트 디렉토리로 이동 후 `pip` 명령을 사용하여 필요한 파이썬 스크립트를 내려받는다.

```bash
cd echeck
```

```bash
pip install -r requirements.txt
```

## 빌드

### libdarknet.so
darknet.py 스크립트를 실행하기 위해선 darknet 레포지토리로부터 소스코드를 내려받아 libdarknet.so 파일을 make로 빌드하여 만들어야 한다.

다음과 같이 [Darknet](https://github.com/AlexeyAB/darknet.git) 으로부터 소스코드 clone 하여 darknet 디렉토리로 내려받는다.

```bash
git clone https://github.com/AlexeyAB/darknet.git -P darknet
cd darknet
```

### darknet 설정
darknet 디렉토리로 이동한 후 `Makefile` 의 상단 내용을 아래와 같이 수정해준다.

```bash
GPU=1
CUDNN=1
CUDNN_HALF=1
...
LIBSO=1
...
```

`make` 명령을 통해 libdarknet.so 파일을 생성해준다.

```bash
# in 'darknet' directory
make -j$(nproc)
```

libdarknet.so 파일이 생성된것을 확인하고 darknet 디렉토리 경로를 환경변수로 저장한다.

```bash
ls ./libdarknet.so
# ./libdarknet.so
export DARKNET_PATH=${PWD}
```

## 모델 다운로드

소스코드의 root 경로로 이동하고 다음과 같이 모델들을 내려받는다.

전자식 숫자 객체 탐지(.weight), 설정 파일과 숫자 분류기 모델(.h5)를 `model` 경로로 내려받는다.

**emeter-yolov4-416.weight :**

- `url` : [http://gofile.me/5RYLE/VqHeQUVYn](http://gofile.me/5RYLE/VqHeQUVYn)
- `save to` : model/num_detector

**LeNet-100.h5 :** 

- `url` : [http://gofile.me/5RYLE/9163idFmh](http://gofile.me/5RYLE/9163idFmh)
- `save to` : model/num_classifier

기계식 전력량계 숫자 탐지(TF-Serving)을 다운로드 받고 압축을 해제한 다음 Docker의 Tensorflow-Serving 이미지를 내려받아 API를 실행시켜준다.

**meter-yolov4-608.zip :**

- `url` : [http://gofile.me/5RYLE/5GYbfW58a](http://gofile.me/5RYLE/5GYbfW58a)
- `save to` : model/meter-yolov4

```bash
# move to model dir and extract zip file
cd model/num_detector
unzip meter-yolov4-608.zip
rm meter-yolov4-608.zip
```

```bash
# pull tf-serving image
docker pull tensorflow/serving:latest

# start meter detector model with tf-serving on docker
docker run --name meter-608 -d -p 8501:8501 \
--mount type=bind,source=${PWD}/meter-yolov4-608,target=/models/meter-608 \
-e MODEL_NAME=meter-608 -t tensorflow/serving:latest
```

## 실행

80 포트로 서비스하기 때문에 **OS가 리눅스인 경우 sudo** 를 붙여서 관리자 권한으로 실행해야한다.

```bash
# if not root
sudo python api.py
```
