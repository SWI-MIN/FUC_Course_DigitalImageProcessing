import cv2
import os
import numpy as np

def read_img(img_path):     # 讀檔, input = 影像路徑, output = 圖像,檔名,副檔名
    img_filepath = os.path.splitext(img_path)[0]    # 拆分路徑 & 副檔名，0 為路徑
    img_fileextension = os.path.splitext(img_path)[1]  # 1 為副檔名
    img_filename = os.path.basename(img_filepath)     # 取出檔名不含副檔名
    img = cv2.imread(img_path)  # 讀檔(中文路徑會爆掉)
    return img, img_filename, img_fileextension

def SobelFilter(img):
    img = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)  # img轉灰階再做高斯模糊
    gradient = np.zeros(img.shape)       # 存梯度的大小，梯度方向上的改變率
    G_x = np.zeros(img.shape)             # x 方向梯度
    G_y = np.zeros(img.shape)             # y 方向梯度
    kernel_x = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))  # gradient operators(梯度運算子) sobel (gaussian-like smoothing)
    kernel_y = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
    size = img.shape
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            G_x[i, j] = np.sum(np.multiply(img[i - 1 : i + 2, j - 1 : j + 2], kernel_x))  # 矩陣相乘相加存回中心點
            G_y[i, j] = np.sum(np.multiply(img[i - 1 : i + 2, j - 1 : j + 2], kernel_y))  # python 含前不含後所以加2
    angles = np.rad2deg(np.arctan2(G_y, G_x))  # 梯度方向 = atan(y/x)(簡報P54)，因為np.arctan2回傳的是弧度，因此要將弧度轉化為角度
    angles[angles < 0] += 180   # arctangent定義域[-pi/2,pi/2]    
    gradient = abs(G_x) + abs(G_y) 

    return gradient, angles  # 回傳梯度以及角度
            
def non_maximum_suppression(img, angles):  # 非最大值響應，用以去除假的邊緣響應
    size = img.shape
    suppressed = np.zeros(size)
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            # 依梯度方向(法向量方向):水平、垂直、+-45度
            if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                value_to_compare = max(img[i, j - 1], img[i, j + 1])
            elif (22.5 <= angles[i, j] < 67.5):
                value_to_compare = max(img[i - 1, j - 1], img[i + 1, j + 1])
            elif (67.5 <= angles[i, j] < 112.5):
                value_to_compare = max(img[i - 1, j], img[i + 1, j])
            else:
                value_to_compare = max(img[i + 1, j - 1], img[i - 1, j + 1])
            # 對於某一點P，若他的梯度值沒有比梯度方向兩側點大則其梯度值設為0
            # 此處則為若該處為最大值，才將其填入新圖中
            if img[i, j] >= value_to_compare:
                suppressed[i, j] = img[i, j]
    return suppressed

def double_threshold_hysteresis(img, low, high): # 雙門檻值 介於low, high之間其8連通若有強項素其為邊緣
    size = img.shape
    low_x, low_y = np.where((img < low))
    img[low_x, low_y] = 0
    weak_x, weak_y = np.where((img >= low) & (img < high))
    img[weak_x, weak_y] = 1
    strong_x, strong_y = np.where(img >= high)
    img[strong_x, strong_y] = 2

    def recursion(img, c):  
        i,j = c    
        # dx、dy 分別是周圍8個點的位置
        dx = [-1,-1,-1,0,0,1,1,1]
        dy = [-1,0,1,-1,1,-1,0,1]

        for x,y in zip(dx,dy):
            if img[i + x,j + y] == 1:
                # 如果有弱像素存在把那個點設成強像素、再移動到那個點、用遞廻不斷查看下一個點
                img[i + x,j + y] = 2
                recursion(img,(i + x,j + y))
                
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            if img[i,j] == 2:
                recursion(img, (i, j))
    img = np.where(img == 2,255,0)
    return np.uint8(img)

def Canny(img, low, high):
    img, angles = SobelFilter(img)
    # cv2.imshow('SobelFilter', img)
    # cv2.imwrite('./Test_Img/' + name + '_SobelFilter.jpg',img)
    gradient = np.copy(img)
    img = non_maximum_suppression(img, angles)
    # cv2.imshow('non_maximum_suppression', img)
    # cv2.imwrite('./Test_Img/' + name + '_non_maximum_suppression.jpg',img)
    img = double_threshold_hysteresis(img, low, high)
    return img, gradient

def HoughLinesP(canny):  # 定義Hough轉換參數
    HoughLinesP = canny.copy()
    lines = cv2.HoughLinesP(canny, rho=1, theta=np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10)
    for line in [l[0] for l in lines]:  # 畫線
        leftx, boty, rightx, topy = line
        cv2.line(HoughLinesP, (leftx, boty), (rightx,topy), (255, 255, 0), 2)
    return HoughLinesP

def combine_images(foreground, background, start_y, start_x):
    foreground=cv2.addWeighted(background[start_y:start_y+foreground.shape[0], start_x:start_x+foreground.shape[1]],1,foreground,1,0)
    background[start_y:start_y+foreground.shape[0], start_x:start_x+foreground.shape[1]] = foreground
    return background

img, name, extension = read_img('E:/Program_File/PYTHON/Digital_Image_Processing_HomeWork/HW_3/Test_Img/01.jpg')

canny, gradient = Canny(img, 50, 150)
HoughLinesP = HoughLinesP(canny)
cv2.imshow('Canny', canny)
# cv2.imwrite('./Test_Img/' + name + '_canny.jpg',canny)
cv2.imshow('HoughLinesP', HoughLinesP)
# cv2.imwrite('./Test_Img/' + name + '_HoughLinesP.jpg',HoughLinesP)


img_canny_signature, img_canny_signature_name, img_canny_signature_extension  = \
    read_img('E:/Program_File/PYTHON/Digital_Image_Processing_HomeWork/HW_3/Test_Img/01_canny.jpg')  # 讀檔(中文路徑會爆掉)
img_HoughLinesP_signature, img_HoughLinesP_signature_name, img_HoughLinesP_signature_extension  = \
    read_img('E:/Program_File/PYTHON/Digital_Image_Processing_HomeWork/HW_3/Test_Img/01_HoughLinesP.jpg')  # 讀檔(中文路徑會爆掉)
signature, name_signature, extension_signature = \
    read_img('E:/Program_File/PYTHON/Digital_Image_Processing_HomeWork/HW_3/Test_Img/SWIMIN.png')

canny_signature = combine_images(signature, img_canny_signature, 20, 20)  # 呼叫上面def的副函式，第一個參數是前景，第二個為背景，第三四個為開始位置 Y*X (Y:直的，X:橫的)
HoughLinesP_signature = combine_images(signature, img_HoughLinesP_signature, 20, 20)  # 呼叫上面def的副函式，第一個參數是前景，第二個為背景，第三四個為開始位置 Y*X (Y:直的，X:橫的)
cv2.imshow('canny_signature image', canny_signature)
# cv2.imwrite('./Test_Img/' + img_canny_signature_name + '_canny_signature.jpg',canny_signature)
cv2.imshow('HoughLinesP_signature image', HoughLinesP_signature)
# cv2.imwrite('./Test_Img/' + img_HoughLinesP_signature_name + '_HoughLinesP_signature.jpg',HoughLinesP_signature)

cv2.waitKey(0)
cv2.destroyAllWindows()


