import os
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt
def rgb2hsv(img):
    r,g,b = cv2.split(img)      # 分離RGB
    r, g, b = r/255.0, g/255.0, b/255.0
    # HSV 分別承接經過公式轉換後的H,S,V值    # 高(直行)img.shape[0]     # 寬(橫列)img.shape[1]   # \ 多行語句
    H, S, V = np.zeros((img.shape[0], img.shape[1]), np.float32), \
        np.zeros((img.shape[0], img.shape[1]), np.float32),  np.zeros((img.shape[0], img.shape[1]), np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            max_ = max((b[i, j], g[i, j], r[i, j]))
            min_ = min((b[i, j], g[i, j], r[i, j]))
            # H
            if max_ == min_:           # 當 max = min 時，相位角為'0度'
                H[i, j] = 0
            elif max_ == r[i, j]:      # 當 r  為 max 時，相位角為'60度'
                if g[i, j] >= b[i, j]:      # 當 g >= b 時，+0
                    H[i, j] = (60 * ((g[i, j]) - b[i, j]) / (max_ - min_))
                else:                       # 當 g < b 時，+360
                    H[i, j] = (60 * ((g[i, j]) - b[i, j]) / (max_ - min_))+360
            elif max_ == g[i, j]:       # 當 g 為 max 時，相位角為'60度' + 120
                H[i, j] = 60 * ((b[i, j]) - r[i, j]) / (max_ - min_) + 120
            elif max_ == b[i, j]:       # 當 b 為 max 時，相位角為'60度' + 240
                H[i, j] = 60 * ((r[i, j]) - g[i, j]) / (max_ - min_)+ 240
            # hsv 中的值是 0-360, 0-1, 0-1，但是在 openCV 中 hsv 的值是 0-180, 0-255, 0-255
            H[i,j] =int( H[i,j] / 2)
            # S
            if max_ == 0:
                S[i, j] = 0
            else:
                S[i, j] =int( (max_ - min_)/max_*255)
            # V
            V[i, j] =int( max_*255)
            # print(H[i, j],S[i, j],V[i, j])
    # 合併輸出
    # HSV = cv2.merge([H,S,V])
    # HSV=np.array(HSV,dtype='uint8')
    # return HSV
    # 不合併輸出
    return H, S, V

def hsv_2_bi():
    img_path='E:/Program_File/PYTHON/數位影像處理作業/HW_2/hand7.jpg'
    img_filepath = os.path.splitext(img_path)[0]    # 拆分路徑 & 副檔名，0 為路徑
    img_fileextension = os.path.splitext(img_path)[1]  # 1 為副檔名
    img_filename = os.path.basename(img_filepath)     # 取出檔名不含副檔名
    img = cv2.imread(img_filename + img_fileextension,-1)  # 讀檔


    h, s, v  = rgb2hsv(img)
    bi_img_hav = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if 116 > h[i, j] and 80 < h[i, j] \
                and 256 > s[i, j] and 48 < s[i, j] \
                and 256 > v[i, j] and 50 < v[i, j]:
                bi_img_hav[i, j] = 255
            else:
                bi_img_hav[i, j] = 0

    hsv = cv2.merge([h, s, v])      # 合併HSV
    hsv=np.array(hsv,dtype='uint8')
    cv2.imshow('img_00',hsv)        # HSV SHOW
    bi_img_hav = np.array(bi_img_hav,dtype='uint8')
    cv2.imshow('img_01',bi_img_hav)

    cv2.imwrite(img_filename+'_hsv'+img_fileextension,hsv)# 寫檔
    cv2.imwrite(img_filename+'hand5_hsv_bi'+img_fileextension,bi_img_hav)# 寫檔

    cv2.waitKey()

def rgb_2_bi():
    img_path='E:/Program_File/PYTHON/數位影像處理作業/HW_2/hand7.jpg'
    img_filepath = os.path.splitext(img_path)[0]    # 拆分路徑 & 副檔名，0 為路徑
    img_fileextension = os.path.splitext(img_path)[1]  # 1 為副檔名
    img_filename = os.path.basename(img_filepath)     # 取出檔名不含副檔名
    img = cv2.imread(img_filename + img_fileextension,-1)  # 讀檔

    b,g,r = cv2.split(img)      # 分離RGB
    bi_img_rgb = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            max_ = max((b[i, j], g[i, j], r[i, j]))
            min_ = min((b[i, j], g[i, j], r[i, j]))
            if r[i, j] > 95 and b[i, j] > 20 and (max_ - min_) > 15 and r[i, j] > g[i, j] and r[i, j] > b[i, j]:
                bi_img_rgb[i, j] = 255
            else:
                bi_img_rgb[i, j] = 0

    rgb = cv2.merge([r, g, b])      # bgr 改用 rgb 存
    rgb=np.array(rgb,dtype='uint8')
    cv2.imshow('img_02',rgb)        # rgb SHOW
    bi_img_rgb = np.array(bi_img_rgb,dtype='uint8')
    cv2.imshow('img_03',bi_img_rgb)

    cv2.imwrite(img_filename+'_rgb'+img_fileextension,rgb)# 寫檔
    cv2.imwrite(img_filename+'hand5_bgr2rgb_bi'+img_fileextension,bi_img_rgb)# 寫檔

    cv2.waitKey()  


hsv_2_bi()
rgb_2_bi()

# RGB 範圍 190 138 117    250 205 179
#          17  38.4 74.5  22 28.4 98

# 色彩範圍
# https://blog.csdn.net/wanggsx918/article/details/23272669
# 手勢識別（一）：獲取圖像並進行膚色檢測（Python）
# https://www.twblogs.net/a/5ca72320bd9eee5b1a07541d