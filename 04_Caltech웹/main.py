import enum
from unicodedata import decimal
from flask import Flask, render_template, request, Response
import os
 
import pickle
import pandas as pd
import numpy as np
 
from tensorflow.keras.models import load_model
 
from PIL import Image
 
# opencv
# pip install opencv-python
import cv2
 
# template_folder : html 파일이 모여있는 곳
# static_folder : 이미지, 동영상, 사운드, js, css 파일 등등
app = Flask(__name__, template_folder='view', static_folder='upload')
 
@app.route('/')
def index() :
    html = render_template('index.html')
    return html
 
@app.route('/cnn')
def cnn() :
    html = render_template('cnn.html')
    return html
 
@app.route('/cnn_result', methods=['post'])
def cnn_result() :
   
    if 'upload' in request.files :
        # 업로드된 이미지를 저장한다.
        file_data = request.files['upload']
        file_data.save(f'upload/{file_data.filename}')
 
        # 학습모델을 복원한다.
        model = load_model('model/imageCNN.h5')
 
        # 결과 데이터 이름을 복원한다.
        with open('data/Caltech_classes.dat', 'rb') as fp :
            categories = pickle.load(fp)
 
        # print(model)
        # print(categories)
 
        # 이미지 전처리
 
        # 이미지의 크기
        # 이미지를 모두 같은 사이즈로 조정해야 한다.
        image_w = 64
        image_h = 64
 
        # 이미지  변형작업
        # 이미지 데이터들을 담을 리스트
        X = []
 
        # 이미지를 읽어온다.
        img = Image.open(f'upload/{file_data.filename}')
        # 이미지 파일은 GBR 형태로 되어 있다.
        # CNN을 사용할 것이기 때문에 RGB 형태로 변환한다.
        img = img.convert('RGB')
        # 이미지의 크기를 조정한다.
        img = img.resize((image_h, image_w))
        # 행렬로 변환한다.
        data = np.array(img)
       
        # 리스트에 담는다.
        X.append(data)
 
        X = np.array(X)
 
        # 예측 결과를 가져온다.
        pred = model.predict(X)
        # 각 예측 결과 중 가장 큰 값을 가지고 있는 곳의 인덱스를 가져온다.
        result = np.argmax(pred, axis=1)
        # print('----------------------------------'+ categories[result[0]])
        result2 = categories[result[0]]
 
    return render_template('cnn_result.html', name = result2, filename=file_data.filename)
 
@app.route('/opencv_image')
def opencv_image() :
    html = render_template('opencv_image.html')
    return html
 
@app.route('/opencv_image_result', methods=['post'])
def opencv_image_result() :
    if 'upload' in request.files :
        # 업로드된 이미지를 저장한다.
        file_data = request.files['upload']
        file_data.save(f'upload/{file_data.filename}')
 
        # 신경망 모델 구조 파일
        cfg_file = 'yolo/yolov3.cfg'
        # 가중치 파일
        weights_file = 'yolo/yolov3.weights'
        # 인식 가능한 사물의 이름
        class_file = 'yolo/coco.names'
 
        # yolo 모델을 생성한다.
        model = cv2.dnn.readNet(weights_file, cfg_file)
        # print(model)
 
        # 물체 종류 리스트를 추출한다.
        with open(class_file, 'rt') as fp :
            classes = fp.readlines()
 
        for idx, a1 in enumerate(classes) :
            classes[idx] = classes[idx].strip()
       
        # print(classes)
 
        # 각 사물의 색상값을 랜덤하게 추출한다.
        # 0 ~ 255까지 총 len(classes)만큼의 행이 생기고 각 행은 3개씩 구성된다.
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        # print(colors)
 
        # 사용할 이미지를 불러온다.
        img = cv2.imread(f'upload/{file_data.filename}')
 
        # 이미지의 세로, 가로, 채널 정보를 가지고 온다.
        height, width, channel = img.shape
 
        # 이미지의 크기를 조종한다.
        # 16 X 16 부터해서 16의 배수로 설정해주면 된다.
        a1 = cv2.resize(img, (416, 416))
 
        # 2진 데이터로 변환을 한다.
        # opencv는 주어진 이미지 색상 데이터를 공식을 가지고 계산하여
        # 변환을 한다. 변환된 값에 대해 설정한 임계값 보다 작으면 설정한
        # 색상 값으로 변환을 한다.
        # 첫 번째 : 이미지데이터
        # 두 번째 : 임계값. opencv에서 정해준 값
        # 세 번째 : 이미지 데이터의 크기
        # 네 번째 : 임계값보다 작게 계산된 이미지의 부분을 채워줄 색상 값
        # 결론은 검출되지 않는 부분들은 모두 0, 0, 0(검정색)으로 변환하는 과정
        blob = cv2.dnn.blobFromImage(a1, 0.00392, (416, 416), (0, 0, 0))
 
        # 출력층의 이름을 가져온다.
        outs = model.getUnconnectedOutLayersNames()
        # print(outs)
 
        # 데이터를 학습 모델에 넣어준다.
        model.setInput(blob)
 
        # 물체를 검출한다.
        results = model.forward(outs)
 
        # 예측된 결과를 담을 리스트
        class_id_list = []
        # 예측 정확도
        confidence_list = []
        # 인지된 사물의 영역
        box_list = []
        # 확률의 임계값
        # 확률이 이 값 이상인 것들만 담는다.
        confidence_limit = 0.1
 
        # 출력층의 수 만큼 반복한다.
        # 예측결과 : 앞에서 4개는 검출한 사물의 위치, 5번째는 확률, 그 이후는
        # 원핫 인코딩된 예측 결과이다.
        for out in results :
            # 현재의 출력층이 검출한 사물의 수 만큼 반복한다.
            for detection in out :
                # 원핫 인코딩된 결과를 가져온다.
                score_list = detection[5:]
                # 결과값으로 환산한다.
                class_id = np.argmax(score_list)
                # 현재 사물의 확률 값을 구한다.
                confidence = score_list[class_id]
 
                # 구한 확률 값이 확률 임계값 이상인 것만 사용한다.
                if confidence >= confidence_limit :
 
                    # 물체의 좌표를 계산한다.
                    # 중심점 X, 중심점 Y, 가로, 세로
                    # 결과는 비율값으로 나온다.
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
 
                    # 가로길이
                    w = int(detection[2] * width)
                    # 세로길이
                    h = int(detection[3] * height)
                    # 좌측상단 X
                    x = int(center_x - w / 2)
                    # 좌측상단 y
                    y = int(center_y - h / 2)
 
                    # print(class_id)
 
                    # 담는다.
                    box_list.append([x, y, w, h])
                    confidence_list.append(float(confidence))
                    class_id_list.append(class_id)
 
        # IoU : 두 면적이 중첩되는 영역의 넓이를 두 면적을 합친 총 면적으로 나눈 값
        # 두 면적이 얼마자 중첩되어 있는지 평가하는 지표가 된다.
        # 0 ~ 1사이가 나오며 값이 클수록 중첩된 부분이 많다고 평가한다.
 
        # NMS : IoU 방식으로 면적을 평가하고 중첩이 많이 되었다고 판단되는 역영을
        # 제거하는 방식. opencv가 제공하는 함수는 IoU 방식으로 평가하여 중첩이 많이
        # 되었다고 판단되는 영역 전체를 아우룰수 있는 영역을 만들어준다.
        # 0.4 : 확인하지 않고 버릴 영역의 IoU 값
        indexes = cv2.dnn.NMSBoxes(box_list, confidence_list, confidence_limit, 0.4)
 
        # 검출된 물체에 네모를 그린다.
        for i in range(len(box_list)) :
 
            # NMS를 통해 추출한 영역만 그린다.
            if i in indexes :
 
                # 좌표를 추출한다.
                x, y, w, h = box_list[i]
                # 이름을 추출한다.
                idx = class_id_list[i]
                label = classes[idx]
                # 색상
                color = colors[idx]
 
                # 네모를 그린다.
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                # 이름을 표시한다.
                cv2.putText(img, label, (x, y - 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)
 
        # 이미지 파일로 저장한다.
        cv2.imwrite('upload/yolo_image_result.jpg', img)
 
 
    return render_template('opencv_image_result.html')
 
@app.route('/opencv_video')
def opencv_video() :
    html = render_template('opencv_video.html')
    return html
 
@app.route('/opencv_video_result', methods=['post'])
def opencv_video_result() :
    if 'upload' in request.files :
        # 업로드된 영상을 저장한다.
        file_data = request.files['upload']
        file_data.save(f'upload/yolo_video.mp4')
 
    html = render_template('yolo_video_result.html')
    return html

@app.route('/yolo_streamming')
def yolo_streamming() :
    # mimetype : 데이터의 형태를 알려주는 문자열
    m1 = 'multipart/x-mixed-replace; boundary=frame'
    # 프레임 데이터를 받아온다.
    frame = yolo_video_detecting()
    # 응답객체를 만들어준다.
    r1 = Response(frame, mimetype=m1)
 
    return r1
 
# 영상에서 프레임을 추출하여 반환하는 함수
def yolo_video_detecting()  :

    # 신경망 모델 구조 파일
    cfg_file = 'yolo/yolov3.cfg'
    # 가중치 파일
    weights_file = 'yolo/yolov3.weights'
    # 인식 가능한 사물의 이름
    class_file = 'yolo/coco.names'

    # yolo 모델을 생성한다.
    model = cv2.dnn.readNet(weights_file, cfg_file)
    # print(model)

    # 물체 종류 리스트를 추출한다.
    with open(class_file, 'rt') as fp :
        classes = fp.readlines()

    for idx, a1 in enumerate(classes) :
        classes[idx] = classes[idx].strip()
    
    # print(classes)

    # 각 사물의 색상값을 랜덤하게 추출한다.
    # 0 ~ 255까지 총 len(classes)만큼의 행이 생기고 각 행은 3개씩 구성된다.
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    # print(colors)
 
    # 영상 데이터를 가져온다.
    # cap = cv2.VideoCapture('upload/yolo_video.mp4')
    cap = cv2.VideoCapture(0) # 이거하면 웹캠임
 
    while True :
        # 현재의 프레임을 가져온다.
        ret, img = cap.read()
        # 더이상 가져온 것이 없다면(끝까지 갔다면)
        if img is None :
            break

        # 이미지의 세로, 가로, 채널 정보를 가지고 온다.
        height, width, channel = img.shape
 
        # 이미지의 크기를 조종한다.
        # 16 X 16 부터해서 16의 배수로 설정해주면 된다.
        a1 = cv2.resize(img, (416, 416))
 
        # 2진 데이터로 변환을 한다.
        # opencv는 주어진 이미지 색상 데이터를 공식을 가지고 계산하여
        # 변환을 한다. 변환된 값에 대해 설정한 임계값 보다 작으면 설정한
        # 색상 값으로 변환을 한다.
        # 첫 번째 : 이미지데이터
        # 두 번째 : 임계값. opencv에서 정해준 값
        # 세 번째 : 이미지 데이터의 크기
        # 네 번째 : 임계값보다 작게 계산된 이미지의 부분을 채워줄 색상 값
        # 결론은 검출되지 않는 부분들은 모두 0, 0, 0(검정색)으로 변환하는 과정
        blob = cv2.dnn.blobFromImage(a1, 0.00392, (416, 416), (0, 0, 0))
 
        # 출력층의 이름을 가져온다.
        outs = model.getUnconnectedOutLayersNames()
        # print(outs)
 
        # 데이터를 학습 모델에 넣어준다.
        model.setInput(blob)
 
        # 물체를 검출한다.
        results = model.forward(outs)
 
        # 예측된 결과를 담을 리스트
        class_id_list = []
        # 예측 정확도
        confidence_list = []
        # 인지된 사물의 영역
        box_list = []
        # 확률의 임계값
        # 확률이 이 값 이상인 것들만 담는다.
        confidence_limit = 0.1
 
        # 출력층의 수 만큼 반복한다.
        # 예측결과 : 앞에서 4개는 검출한 사물의 위치, 5번째는 확률, 그 이후는
        # 원핫 인코딩된 예측 결과이다.
        for out in results :
            # 현재의 출력층이 검출한 사물의 수 만큼 반복한다.
            for detection in out :
                # 원핫 인코딩된 결과를 가져온다.
                score_list = detection[5:]
                # 결과값으로 환산한다.
                class_id = np.argmax(score_list)
                # 현재 사물의 확률 값을 구한다.
                confidence = score_list[class_id]
 
                # 구한 확률 값이 확률 임계값 이상인 것만 사용한다.
                if confidence >= confidence_limit :
 
                    # 물체의 좌표를 계산한다.
                    # 중심점 X, 중심점 Y, 가로, 세로
                    # 결과는 비율값으로 나온다.
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
 
                    # 가로길이
                    w = int(detection[2] * width)
                    # 세로길이
                    h = int(detection[3] * height)
                    # 좌측상단 X
                    x = int(center_x - w / 2)
                    # 좌측상단 y
                    y = int(center_y - h / 2)
 
                    # print(class_id)
 
                    # 담는다.
                    box_list.append([x, y, w, h])
                    confidence_list.append(float(confidence))
                    class_id_list.append(class_id)
 
        # IoU : 두 면적이 중첩되는 영역의 넓이를 두 면적을 합친 총 면적으로 나눈 값
        # 두 면적이 얼마자 중첩되어 있는지 평가하는 지표가 된다.
        # 0 ~ 1사이가 나오며 값이 클수록 중첩된 부분이 많다고 평가한다.
 
        # NMS : IoU 방식으로 면적을 평가하고 중첩이 많이 되었다고 판단되는 역영을
        # 제거하는 방식. opencv가 제공하는 함수는 IoU 방식으로 평가하여 중첩이 많이
        # 되었다고 판단되는 영역 전체를 아우룰수 있는 영역을 만들어준다.
        # 0.4 : 확인하지 않고 버릴 영역의 IoU 값
        indexes = cv2.dnn.NMSBoxes(box_list, confidence_list, confidence_limit, 0.4)
 
        # 검출된 물체에 네모를 그린다.
        for i in range(len(box_list)) :
 
            # NMS를 통해 추출한 영역만 그린다.
            if i in indexes :
 
                # 좌표를 추출한다.
                x, y, w, h = box_list[i]
                # 이름을 추출한다.
                idx = class_id_list[i]
                label = classes[idx]
                # 색상
                color = colors[idx]
 
                # 네모를 그린다.
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                # 이름을 표시한다.
                cv2.putText(img, label, (x, y - 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)
 
        # 웹 브라저가 인식할 수 있는 형태로 변환한다.
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
 
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
 
    cap.release()
 
app.run(debug=True, host='0.0.0.0')