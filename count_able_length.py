import cv2
import os
import torch
import yaml
from PIL import Image
import numpy as np
import pandas as pd

def mid_convert(mid,mid2,mid3,df):
    if mid2 == []:
        for i in range(len(df)):
            xmin = df.iloc[i]['xmin']
            xmax = df.iloc[i]['xmax']
            ymin = df.iloc[i]['ymin']
            ymax = df.iloc[i]['ymax']
            # confidence = df.iloc[i]['confidence']
            xmid = (xmax+xmin)/2
            ymid = (ymin+ymax)/2
            mid2.append([xmid,ymid])
            # mid.append([xmid,ymid,confidence])
        mid2.sort()
    else:
        mid = mid2
        mid2 = []
        for i in range(len(df)):
            xmin = df.iloc[i]['xmin']
            xmax = df.iloc[i]['xmax']
            ymin = df.iloc[i]['ymin']
            ymax = df.iloc[i]['ymax']
            # confidence = df.iloc[i]['confidence']
            xmid = (xmax+xmin)/2
            ymid = (ymin+ymax)/2
            mid2.append([xmid,ymid])
            # mid.append([xmid,ymid,confidence])
        mid2.sort()
        mid3 = []
        
        for middle in mid:
            for middle2 in mid2:
                # mid_x , mid_y ,_ = middle
                # mid2_x, mid2_y ,_ = middle2
                mid_x , mid_y = middle
                mid2_x, mid2_y = middle2
                if (mid_x*(1-threshold) < mid2_x < mid_x*(1+threshold)) & (mid_y*(1-threshold) < mid2_y < mid_y*(1+threshold)):
                    mid3.append([mid2_x,mid2_y])
    mid3.sort()
    return mid, mid2, mid3


def able_length():
    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l')  # or yolov5n - yolov5x6, custom
    cctv_pre = pd.read_csv('./cctv_pre.csv',index_col=0,encoding='cp949')

    path = './dataset/'
    file_name = "3-sample_cctv2.mp4"
    filePath = os.path.join(path, file_name)
    cctv_num = int(file_name[0])

    mid = []
    mid2 = []
    mid3 = []
    w_mid = []
    w_mid2 = []
    w_mid3 = []


    coor_df = cctv_pre[cctv_pre.cctv_num == cctv_num]

    count = 0
    threshold = 0.1
    line_width = 15
    line_pixel = coor_df['line_pixel'].values[0]

    print(filePath)

    if os.path.isfile(filePath):	# 해당 파일이 있는지 확인
        # 영상 객체(파일) 가져오기
        cap = cv2.VideoCapture(filePath)
    else:
        print("파일이 존재하지 않습니다.")  

    # 프레임을 정수형으로 형 변환
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))	# 영상의 넓이(가로) 프레임
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))	# 영상의 높이(세로) 프레임
    
    frame_size = (frameWidth, frameHeight)
    print('frame_size={}'.format(frame_size))

    frameRate = 30
    
    while True:
        # 한 장의 이미지(frame)를 가져오기
        # 영상 : 이미지(프레임)의 연속
        # 정상적으로 읽어왔는지 -> retval
        # 읽어온 프레임 -> frame
        retval, frame = cap.read()
        if not(retval):	# 프레임정보를 정상적으로 읽지 못하면
            break  # while문을 빠져나가기
            
        key = cv2.waitKey(frameRate)  # frameRate msec동안 한 프레임을 보여준다
        count += 1
        # print(count)
        
        # 60프레임마다 객체 탐지 작동
        if count % 60 == 0:
            results = model(frame)
            # results.show()
            # Results
            results.pandas().xyxy[0] 
            df = results.pandas().xyxy[0][results.pandas().xyxy[0]['class']==2]
            mid, mid2, mid3 = mid_convert(mid, mid2, mid3,df)
            # print(mid)
            # print()
            # print(mid2)
            # print()
            # print(mid3)
            # print(mid2==mid3)
            if mid2==mid3:
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
                results = model(result)
                # Results
                results.save()
                results.pandas().xyxy[0]  # or .show(), .save(), .crop(), .pandas(), etc.
                df = results.pandas().xyxy[0][results.pandas().xyxy[0]['class']==2]
                df = df.reset_index(drop=True)
                w_mid, w_mid2, w_mid3 = mid_convert(w_mid, w_mid2, w_mid3,df)

                if w_mid2 == w_mid3:
                    left = []
                    right = []
                    cross = []
                    print('-'*30)
                    df['xmid'] = np.NaN
                    for i in range(len(df)):
                        xmin = df.loc[i]['xmin']
                        xmax = df.loc[i]['xmax']
                        xmid = (xmax+xmin)/2
                        df.loc[i,['xmid']] = xmid
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

                    able_length = x_length / line_pixel
                    print(able_length*line_width) # 출력값 = 통행가용폭

    return able_length*line_width