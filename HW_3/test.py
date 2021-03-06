import cv2
import os
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def read_img(img_path):     # 讀檔, input = 影像路徑, output = 圖像,檔名,副檔名
    img_filepath = os.path.splitext(img_path)[0]    # 拆分路徑 & 副檔名，0 為路徑
    img_fileextension = os.path.splitext(img_path)[1]  # 1 為副檔名
    img_filename = os.path.basename(img_filepath)     # 取出檔名不含副檔名
    # img = cv2.imread(img_path, -1)  # 讀檔(中文路徑會爆掉)
    img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1) # 讀檔(含有中文路徑)
    return img, img_filename, img_fileextension

def Grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def GaussianBlur(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

def SobelFilter(img):
    img = GaussianBlur(Grayscale(img))  # img轉灰階再做高斯模糊
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

    # 平方開根號與取絕對值效果差不多，為求運算效率，有時會用絕對值取得近似值
    # gradient = abs(G_x) + abs(G_y) 
    gradient = np.sqrt(np.square(G_x) + np.square(G_y))
    # gradient = np.multiply(gradient, 255.0 / gradient.max())  # 255.0 / gradient.max()將gradient轉成0-1，在乘回gradient就會變成0-255
    # gradient = gradient.astype('uint8')  # 將scale轉換成8-bit(簡報P55)

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

# def combine(img,c):
#     i,j = c
#     # dx、dy 分別是周圍8個點的位置
#     dx = [-1,-1,-1,0,0,1,1,1]
#     dy = [-1,0,1,-1,1,-1,0,1]

#     for x,y in zip(dx,dy):
#         if img[i + x,j + y] == 1:
#             # 如果有弱像素存在把那個點設成強像素、再移動到那個點、用遞廻不斷查看下一個點
#             img[i + x,j + y] = 2
#             combine(img,(i + x,j + y))
# def double_threshold_hysteresis(img,low,high):
#     size = img.shape
#     # 加果是強像素設成2、弱像素設1、其他設0
#     for i in range(1, size[0] - 1):
#         for j in range(1, size[1] - 1):
#             if img[i,j] >= high:
#                 img[i,j] = 2
#             elif high > img[i,j] >= low:
#                 img[i,j] = 1
#             else:
#                 img[i,j] = 0

#     for i in range(1,size[0] - 1):
#         for j in range(1,size[1] - 1):
#             if img[i,j] == 2:
#                 # 確認強像素周圍的點
#                 combine(img,(i,j))
    
#     img = np.where(img == 2,255,0)
#     return np.uint8(img)
    
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
# 雙門檻值，大於high為強像素，小於low為弱像素，介於兩者之間其周圍4連通或8連通若有強項素其為邊緣
def double_threshold_hysteresis(img, low, high):  
    size = img.shape
    low_x, low_y = np.where((img < low))
    img[low_x, low_y] = 0
    weak_x, weak_y = np.where((img >= low) & (img < high))
    img[weak_x, weak_y] = 1
    strong_x, strong_y = np.where(img >= high)
    img[strong_x, strong_y] = 2
    # 不知道怎麼做，最好的方法應該是從強像素向外，如過世若像素將其轉為強像素
    
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            if img[i,j] == 2:
                recursion(img, (i, j))
    img = np.where(img == 2,255,0)
    return np.uint8(img)



def Canny(img, low, high):
    img, angles = SobelFilter(img)
    # cv2.imshow('SobelFilter', img)
    # cv2.imwrite('./Test_Img/' + file_name + '_SobelFilter.jpg',img)
    gradient = np.copy(img)
    img = non_maximum_suppression(img, angles)
    # cv2.imshow('non_maximum_suppression', img)
    # cv2.imwrite('./Test_Img/' + file_name + '_non_maximum_suppression.jpg',img)
    img = double_threshold_hysteresis(img, low, high)
    # cv2.imshow('double_threshold_hysteresis', img)
    # cv2.imwrite('./Test_Img/' + file_name + '_double_threshold_hysteresis.jpg',img)
    return img, gradient



img, file_name, file_extension = read_img('E:/Program_File/PYTHON/數位影像處理作業/HW_3/Test_Img/03.jpg')
edges, gradient = Canny(img, 50, 150)

cv2.imshow('Canny', edges)
# cv2.imwrite('./Test_Img/' + file_name + '_canny.jpg',edges)
cv2.waitKey(0)
cv2.destroyAllWindows()




# Edge-detection---Canny-detector
# https://github.com/StefanPitur/Edge-detection---Canny-detector/blob/master/canny.py
# CANNY EDGE DETECTION
# http://justin-liang.com/tutorials/canny/