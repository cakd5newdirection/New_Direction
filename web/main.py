# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import cv2
import os
import torch
import yaml
from PIL import Image
import numpy as np
import pandas as pd

model = torch.hub.load('ultralytics/yolov5', 'yolov5l')  # or yolov5n - yolov5x6, custom
model2 = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt', autoshape=True)

threshold = 0.1
line_width = 15

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


def mid_convert(mid,mid2,mid3,df,threshold):
    if mid2 == []:
        for i in range(len(df)):
            xmin = df.iloc[i]['xmin']
            xmax = df.iloc[i]['xmax']
            ymin = df.iloc[i]['ymin']
            ymax = df.iloc[i]['ymax']
            confidence = df.iloc[i]['confidence']
            xmid = (xmax+xmin)/2
            ymid = (ymin+ymax)/2
            # mid2.append([xmid,ymid])
            mid2.append([xmid,ymid,confidence])
        mid2.sort()
    else:
        mid = mid2
        mid2 = []
        for i in range(len(df)):
            xmin = df.iloc[i]['xmin']
            xmax = df.iloc[i]['xmax']
            ymin = df.iloc[i]['ymin']
            ymax = df.iloc[i]['ymax']
            confidence = df.iloc[i]['confidence']
            xmid = (xmax+xmin)/2
            ymid = (ymin+ymax)/2
            # mid2.append([xmid,ymid])
            mid2.append([xmid,ymid,confidence])
        mid2.sort()
        mid3 = []
        
        for middle in mid:
            for middle2 in mid2:
                # mid_x , mid_y ,_ = middle
                # mid2_x, mid2_y ,_ = middle2
                mid_x , mid_y, _ = middle
                mid2_x, mid2_y, mid_con2 = middle2
                if (mid_x*(1-threshold) < mid2_x < mid_x*(1+threshold)) & (mid_y*(1-threshold) < mid2_y < mid_y*(1+threshold)):
                    mid3.append([mid2_x,mid2_y,mid_con2])
    mid3.sort()
    return mid, mid2, mid3






app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index_수정보완필요.html")

@app.route("/get")
def able_cal():
    global cap,threshold,line_width,line_pixel,w1_mid,w1_mid2,w1_mid3,able_length,w2_mid,w2_mid2,w2_mid3,w3_mid,w3_mid2,w3_mid3,w4_mid,w4_mid2,w4_mid3
    path = '../web/static'
    file_name = request.args.get('file_name').strip()
    print(file_name)
    filePath = os.path.join(path, file_name)
    cap = cv2.VideoCapture(filePath)
    frame_num = float(request.args.get('dst').strip())
    cctv_num = int(file_name[0])
    cctv_pre = pd.read_csv('../cctv_pre.csv',index_col=0,encoding='cp949')
    coor_df = cctv_pre[cctv_pre.cctv_num == cctv_num]
    line_pixel = coor_df['line_pixel'].values[0]
    able_length = 0

    if frame_num*30 > cap.get(cv2.CAP_PROP_FRAME_COUNT):
        return '0'
    cap.set(cv2.CAP_PROP_POS_FRAMES, (frame_num)*30)
    
    retval, frame = cap.read()
    
        # src = cv2.imread(frame, cv2.IMREAD_COLOR) 
    topLeft = [coor_df.topLeft_x.values[0] , coor_df.topLeft_y.values[0]]
    topRight = [coor_df.topRight_x.values[0] , coor_df.topRight_y.values[0]] 
    bottomRight = [coor_df.bottomRight_x.values[0] , coor_df.bottomRight_y.values[0]] 
    bottomLeft = [coor_df.bottomLeft_x.values[0] , coor_df.bottomLeft_y.values[0]]

    # 변환 전 4개 좌표 
    pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])
    # print(pts1)

    # 변환 후 영상에 사용할 서류의 폭과 높이 계산
    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])
    width = int(max([w1, w2])) # 두 좌우 거리간의 최대값이 서류의 폭
    height = int(max([h1, h2])*1.5)  # 두 상하 거리간의 최대값이 서류의 높이

    # 변환 후 4개 좌표
    pts2 = np.float32([[0, 0], [width - 1, 0],
                        [width - 1, height - 1], [0, height - 1]])
    # print(pts2)
    # print(width,height)

    # 변환 행렬 계산 
    mtrx = cv2.getPerspectiveTransform(pts1, pts2)
    # 원근 변환 적용
    result = cv2.warpPerspective(frame, mtrx, (width, height))

    image1 = result
    
    img_size = image1.shape # 이미지 사이즈 추출
    # Inference
    results = model2(result)
    # Results
    results.save()
    results.pandas().xyxy[0]  # or .show(), .save(), .crop(), .pandas(), etc.
    df = results.pandas().xyxy[0]
    df = df.reset_index(drop=True)
    if cctv_num == 1:
        w1_mid, w1_mid2, w1_mid3 = mid_convert(w1_mid, w1_mid2, w1_mid3,df,threshold)
        if w1_mid3 != []:
            left = []
            right = []
            cross = []
            print('-'*30)
            df['xmid'] = np.NaN
            for i in range(len(df)):
                for j in range(len(w1_mid3)):
                    if w1_mid3[j][2] == df['confidence'][i]:
                        df.loc[i,['xmid']] = w1_mid3[j][0]
            df.dropna(inplace=True)
            # break
            for i in df.xmid:
                if i > img_size[1]/2:
                    right.append(df.loc[df['xmid']==i].values.tolist()[0])
                else:
                    left.append(df.loc[df['xmid']==i].values.tolist()[0])
            if (left != [])&(right != []):
                for i in left:
                    for j in right:
                        if (i[3]>j[1]) & (i[1]<j[3]):
                            cross.append([i[2],j[0]])
                        elif (i[1]>j[3]) & (i[3]<j[1]):
                            cross.append([i[2],j[0]])
                cross_able = []
                if cross != []:
                    for i in cross:
                        cross_able.append(i[1]-i[0])
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

            able_pixel = x_length / line_pixel
            able_length = able_pixel*line_width
            print(able_length) # 출력값 = 통행가용폭
        return str(able_length)
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
            # break
            for i in df.xmid:
                if i > img_size[1]/2:
                    right.append(df.loc[df['xmid']==i].values.tolist()[0])
                else:
                    left.append(df.loc[df['xmid']==i].values.tolist()[0])
            if (left != [])&(right != []):
                for i in left:
                    for j in right:
                        if (i[3]>j[1]) & (i[1]<j[3]):
                            cross.append([i[2],j[0]])
                        elif (i[1]>j[3]) & (i[3]<j[1]):
                            cross.append([i[2],j[0]])
                cross_able = []
                if cross != []:
                    for i in cross:
                        cross_able.append(i[1]-i[0])
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

            able_pixel = x_length / line_pixel
            able_length = able_pixel*line_width
            print(able_length) # 출력값 = 통행가용폭
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
            # break
            for i in df.xmid:
                if i > img_size[1]/2:
                    right.append(df.loc[df['xmid']==i].values.tolist()[0])
                else:
                    left.append(df.loc[df['xmid']==i].values.tolist()[0])
            if (left != [])&(right != []):
                for i in left:
                    for j in right:
                        if (i[3]>j[1]) & (i[1]<j[3]):
                            cross.append([i[2],j[0]])
                        elif (i[1]>j[3]) & (i[3]<j[1]):
                            cross.append([i[2],j[0]])
                cross_able = []
                if cross != []:
                    for i in cross:
                        cross_able.append(i[1]-i[0])
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

            able_pixel = x_length / line_pixel
            able_length = able_pixel*line_width
            print(able_length) # 출력값 = 통행가용폭
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
            # break
            for i in df.xmid:
                if i > img_size[1]/2:
                    right.append(df.loc[df['xmid']==i].values.tolist()[0])
                else:
                    left.append(df.loc[df['xmid']==i].values.tolist()[0])
            if (left != [])&(right != []):
                for i in left:
                    for j in right:
                        if (i[3]>j[1]) & (i[1]<j[3]):
                            cross.append([i[2],j[0]])
                        elif (i[1]>j[3]) & (i[3]<j[1]):
                            cross.append([i[2],j[0]])
                cross_able = []
                if cross != []:
                    for i in cross:
                        cross_able.append(i[1]-i[0])
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

            able_pixel = x_length / line_pixel
            able_length = able_pixel*line_width
            print(able_length) # 출력값 = 통행가용폭
        return str(able_length)
