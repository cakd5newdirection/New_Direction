# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import cv2
import os
import torch
import yaml
from PIL import Image
import numpy as np
import pandas as pd

model2 = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt', autoshape=True) # warping 후 왜곡된 차량에 대해 학습한 weight 모델 사용

# 오차율설정
threshold = 0.1
# 차선의 실제 넓이 설정(현재는 모든 차선이 15cm로 동일해서 미리 선언)
line_width = 15

# 좌표 중심점을 담을 빈 리스트 생성
# 각 cctv별로 비교를 통해 이동여부를 판단하기에 cctv별로 리스트 생성
w1_mid = []
w1_mid2 = []
w1_mid3 = []
w2_mid = []
w2_mid2 = []
w2_mid3 = []
w3_mid = []
w3_mid2 = []
w3_mid3 = []
w4_mid = []
w4_mid2 = []
w4_mid3 = []

# 탐지된 객체의 중심점 정보와 정확도를 반환
def mid_convert(mid,mid2,mid3,df,threshold):
    # 처음 함수가 실행될 경우에는 mid2가 비어있기에 mid를 채워주기만 함
    if mid2 == []:
        for i in range(len(df)):
            xmin = df.iloc[i]['xmin']
            xmax = df.iloc[i]['xmax']
            ymin = df.iloc[i]['ymin']
            ymax = df.iloc[i]['ymax']
            confidence = df.iloc[i]['confidence']
            xmid = (xmax+xmin)/2
            ymid = (ymin+ymax)/2
            mid2.append([xmid,ymid,confidence])
        mid2.sort()
    # mid2가 차있다면 비교를 시작
    else:
        # mid에 mid2의 좌표를 저장
        mid = mid2
        # mid2 초기화
        mid2 = []

        for i in range(len(df)):
            xmin = df.iloc[i]['xmin']
            xmax = df.iloc[i]['xmax']
            ymin = df.iloc[i]['ymin']
            ymax = df.iloc[i]['ymax']
            confidence = df.iloc[i]['confidence']
            xmid = (xmax+xmin)/2
            ymid = (ymin+ymax)/2
            mid2.append([xmid,ymid,confidence])
        mid2.sort()
        mid3 = []
        # 이전 프레임과 현재 프레임의 중심점을 비교하여 움직이고 있는지 판단
        # 움직이지 않았다고 판단된 좌표와 그 정확도를 mid3 리스트에 저장
        for middle in mid:
            for middle2 in mid2:
                mid_x , mid_y, _ = middle
                mid2_x, mid2_y, mid_con2 = middle2
                if (mid_x*(1-threshold) < mid2_x < mid_x*(1+threshold)) & (mid_y*(1-threshold) < mid2_y < mid_y*(1+threshold)):
                    mid3.append([mid2_x,mid2_y,mid_con2])
    mid3.sort()
    return mid, mid2, mid3






app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index_v4.html")

@app.route("/get")
def able_cal():
    global cap,threshold,line_width,line_pixel,w1_mid,w1_mid2,w1_mid3,able_length,w2_mid,w2_mid2,w2_mid3,w3_mid,w3_mid2,w3_mid3,w4_mid,w4_mid2,w4_mid3
    # 영상 데이터 load
    path = '../web/static'
    file_name = request.args.get('file_name').strip()
    print(file_name)
    filePath = os.path.join(path, file_name)
    cap = cv2.VideoCapture(filePath)
    frame_num = float(request.args.get('dst').strip())
    # cctv에 대한 사전 정보 load
    cctv_num = int(file_name[0])
    cctv_pre = pd.read_csv('../cctv_pre.csv',index_col=0,encoding='cp949')
    coor_df = cctv_pre[cctv_pre.cctv_num == cctv_num]
    line_pixel = coor_df['line_pixel'].values[0]
    able_length = 0
    
    # 영상이 끝났다면 0을 반환
    if frame_num*30 > cap.get(cv2.CAP_PROP_FRAME_COUNT):
        return '0'

    # 입력받은 시간으로 영상을 세트
    cap.set(cv2.CAP_PROP_POS_FRAMES, (frame_num)*30)
    retval, frame = cap.read()

    # 사전에 저장된 warping 좌표를 load
    topLeft = [coor_df.topLeft_x.values[0] , coor_df.topLeft_y.values[0]]
    topRight = [coor_df.topRight_x.values[0] , coor_df.topRight_y.values[0]] 
    bottomRight = [coor_df.bottomRight_x.values[0] , coor_df.bottomRight_y.values[0]] 
    bottomLeft = [coor_df.bottomLeft_x.values[0] , coor_df.bottomLeft_y.values[0]]

    # 변환 전 4개 좌표 
    pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

    # 변환 후 영상에 사용할 차량의 폭과 높이 계산
    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])
    width = int(max([w1, w2])) # 두 좌우 거리간의 최대값이 차량의 폭
    height = int(max([h1, h2])*1.5)  # 두 상하 거리간의 최대값이 차량의 높이

    # 변환 후 4개 좌표
    pts2 = np.float32([[0, 0], [width - 1, 0],
                        [width - 1, height - 1], [0, height - 1]])

    # 변환 행렬 계산 
    mtrx = cv2.getPerspectiveTransform(pts1, pts2)
    # 원근 변환 적용
    result = cv2.warpPerspective(frame, mtrx, (width, height))

    image1 = result
    
    img_size = image1.shape # 이미지 사이즈 추출
    
    # 모델 적용
    results = model2(result)
    # 결과값 저장
    results.save()

    # 결과값 좌표들 데이터프레임에 저장
    results.pandas().xyxy[0]
    df = results.pandas().xyxy[0]

    # 데이터프레임 인덱스 초기화
    df = df.reset_index(drop=True)

    # CCTV 번호 별로 처리
    if cctv_num == 1:

        # 중심점 좌표 및 이동여부 갱신
        w1_mid, w1_mid2, w1_mid3 = mid_convert(w1_mid, w1_mid2, w1_mid3,df,threshold)

        # 이동하지 않은 차량을 통해 도로 폭 계산
        if w1_mid3 != []:
            # 왼쪽, 오른족, 겹쳐있는차량을 넣어줄 리스트 초기화
            left = []
            right = []
            cross_able = []
            print('-'*30)
            # 데이터프레임에 x중심점을 넣을 칼럼 생성
            df['xmid'] = np.NaN
            # 데이터프레임에서 mid3와 정확도가 같은 객체에 중심점좌표 입력
            for i in range(len(df)):
                for j in range(len(w1_mid3)):
                    if w1_mid3[j][2] == df['confidence'][i]:
                        df.loc[i,['xmid']] = w1_mid3[j][0]
            
            # 중심점좌표가 비어있는 데이터프레임 행 삭제
            df.dropna(inplace=True)

            # 이미지 중심점을 기준으로 왼쪽에 있는차량과 오른쪽에 있는 차량을 분리
            for i in df.xmid:
                if i > img_size[1]/2:
                    right.append(df.loc[df['xmid']==i].values.tolist()[0])
                else:
                    left.append(df.loc[df['xmid']==i].values.tolist()[0])
            
            # 감지된 차량으로부터 최소의 픽셀 길이를 측정

            # 왼쪽과 오른쪽 모두에 차량이 존재할 경우
            if (left != [])&(right != []):
                for i in left:
                    for j in right:
                        # 차량이 겹쳐 있다면 cross에 두 차량 사이의 픽셀값을 저장
                        if (i[3]>j[1]) & (i[1]<j[3]):
                            cross_able.append(j[0]-i[2])
                        elif (i[1]>j[3]) & (i[3]<j[1]):
                            cross_able.append(j[0]-i[2])
                # cross_able이 비어있지 않다면 리스트의 최소값이 통행가용폭
                if cross_able != []:
                    x_length = min(cross_able)
                # cross_able이 비어있다면 겹쳐있는 차량이 없는것이기에 왼쪽과 오른쪽을 나눠서 가용폭 계산
                else:
                    able = []

                    # 왼쪽에 있는 차량의 경우 가용폭은 (이미지의 x축 길이 - 차량의 최대 x좌표)
                    for i in range(len(left)):
                        able.append(img_size[1] - left[i][2])

                    # 오른쪽에 있는 차량의 경우 가용폭은 (차량의 최소 x좌표)
                    for i in range(len(right)):
                        able.append(right[i][0])
                    x_length = min(able)

            # 왼쪽과 오른쪽중 한곳에만 챠량이 존재하거나 차량이 없을경우
            else:
                able = []
                # 차량이 없을 경우에는 이미지의 사이즈가 가용폭
                if (left == [])&(right == []):
                    able.append(img_size[1])

                # 왼쪽에 있는 차량의 경우 가용폭은 (이미지의 x축 길이 - 차량의 최대 x좌표)
                elif left != []:
                    for i in range(len(left)):
                        able.append(img_size[1] - left[i][2])

                # 오른쪽에 있는 차량의 경우 가용폭은 (차량의 최소 x좌표)
                else:
                    for i in range(len(right)):
                        able.append(right[i][0])
                x_length = min(able)

            # 1픽셀 당 cm 구하기(실제 차선의 길이 / 차선의 이미지상 픽셀)
            pixel_to_cm = line_width/line_pixel

            # 통행가용폭의 픽셀을 1픽셀당 cm로 곱해서 실제 길이 산출
            able_length =  x_length * pixel_to_cm
            print(able_length) # 출력값 = 통행가용폭
        return str(able_length)
        
    # 이하 동일하기에 생략
    elif cctv_num == 2:
        w2_mid, w2_mid2, w2_mid3 = mid_convert(w2_mid, w2_mid2, w2_mid3,df,threshold)
        if w1_mid3 != []:
            left = []
            right = []
            cross = []
            print('-'*30)
            df['xmid'] = np.NaN
            for i in range(len(df)):
                for j in range(len(w2_mid3)):
                    if w2_mid3[j][2] == df['confidence'][i]:
                        df.loc[i,['xmid']] = w2_mid3[j][0]
            df.dropna(inplace=True)
            for i in df.xmid:
                if i > img_size[1]/2:
                    right.append(df.loc[df['xmid']==i].values.tolist()[0])
                else:
                    left.append(df.loc[df['xmid']==i].values.tolist()[0])
            if (left != [])&(right != []):
                for i in left:
                    for j in right:
                        if (i[3]>j[1]) & (i[1]<j[3]):
                            cross_able.append(j[0]-i[2])
                        elif (i[1]>j[3]) & (i[3]<j[1]):
                            cross_able.append(j[0]-i[2])
                if cross_able != []:
                    x_length = min(cross_able)
                else:
                    able = []
                    for i in range(len(left)):
                        able.append(img_size[1] - left[i][2])
                    for i in range(len(right)):
                        able.append(right[i][0])
                    x_length = min(able)
            else:
                able = []
                if left != []:
                    for i in range(len(left)):
                        able.append(img_size[1] - left[i][2])
                else:
                    for i in range(len(right)):
                        able.append(right[i][0])
                x_length = min(able)

            pixel_to_cm = line_width/line_pixel
            able_length =  x_length * pixel_to_cm
            print(able_length)
        return str(able_length)
    elif cctv_num == 3:
        w3_mid, w3_mid2, w3_mid3 = mid_convert(w3_mid, w3_mid2, w3_mid3,df,threshold)
        if w1_mid3 != []:
            left = []
            right = []
            cross = []
            print('-'*30)
            df['xmid'] = np.NaN
            for i in range(len(df)):
                for j in range(len(w3_mid3)):
                    if w3_mid3[j][2] == df['confidence'][i]:
                        df.loc[i,['xmid']] = w3_mid3[j][0]
            df.dropna(inplace=True)
            for i in df.xmid:
                if i > img_size[1]/2:
                    right.append(df.loc[df['xmid']==i].values.tolist()[0])
                else:
                    left.append(df.loc[df['xmid']==i].values.tolist()[0])
            if (left != [])&(right != []):
                for i in left:
                    for j in right:
                        if (i[3]>j[1]) & (i[1]<j[3]):
                            cross_able.append(j[0]-i[2])
                        elif (i[1]>j[3]) & (i[3]<j[1]):
                            cross_able.append(j[0]-i[2])
                if cross_able != []:
                    x_length = min(cross_able)
                else:
                    able = []
                    for i in range(len(left)):
                        able.append(img_size[1] - left[i][2])
                    for i in range(len(right)):
                        able.append(right[i][0])
                    x_length = min(able)
            else:
                able = []
                if left != []:
                    for i in range(len(left)):
                        able.append(img_size[1] - left[i][2])
                else:
                    for i in range(len(right)):
                        able.append(right[i][0])
                x_length = min(able)

            pixel_to_cm = line_width/line_pixel
            able_length =  x_length * pixel_to_cm
            print(able_length)
        return str(able_length)
    elif cctv_num == 4:
        w4_mid, w4_mid2, w4_mid3 = mid_convert(w4_mid, w4_mid2, w4_mid3,df,threshold)
        if w4_mid3 != []:
            left = []
            right = []
            cross = []
            print('-'*30)
            df['xmid'] = np.NaN
            for i in range(len(df)):
                for j in range(len(w4_mid3)):
                    if w4_mid3[j][2] == df['confidence'][i]:
                        df.loc[i,['xmid']] = w4_mid3[j][0]
            df.dropna(inplace=True)
            for i in df.xmid:
                if i > img_size[1]/2:
                    right.append(df.loc[df['xmid']==i].values.tolist()[0])
                else:
                    left.append(df.loc[df['xmid']==i].values.tolist()[0])
            if (left != [])&(right != []):
                for i in left:
                    for j in right:
                        if (i[3]>j[1]) & (i[1]<j[3]):
                            cross_able.append(j[0]-i[2])
                        elif (i[1]>j[3]) & (i[3]<j[1]):
                            cross_able.append(j[0]-i[2])
                if cross_able != []:
                    x_length = min(cross_able)
                else:
                    able = []
                    for i in range(len(left)):
                        able.append(img_size[1] - left[i][2])
                    for i in range(len(right)):
                        able.append(right[i][0])
                    x_length = min(able)
            else:
                able = []
                if left != []:
                    for i in range(len(left)):
                        able.append(img_size[1] - left[i][2])
                else:
                    for i in range(len(right)):
                        able.append(right[i][0])
                x_length = min(able)

            pixel_to_cm = line_width/line_pixel
            able_length =  x_length * pixel_to_cm
            print(able_length)
        return str(able_length)
