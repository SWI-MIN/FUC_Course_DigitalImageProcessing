import cv2
import os
import numpy as np

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
    # 平方開根號與取絕對值效果差不多，為求運算效率，有時會用絕對值取得近似值
    # gradient = abs(G_x) + abs(G_y) 
    gradient = np.sqrt(np.square(G_x) + np.square(G_y))
    # 不知道這步驟要幹嘛的，不做這步驟會變得有很多多餘的線條，看起來這步驟像是將原先陣列乘上
    gradient = np.multiply(gradient, 255.0 / gradient.max())  

    angles = np.rad2deg(np.arctan2(G_y, G_x))  # 梯度方向 = atan(y/x)(簡報P54)，因為np.arctan2回傳的是弧度，因此要將弧度轉化為角度
    angles[angles < 0] += 180   # arctangent定義域[-pi/2,pi/2]
    gradient = gradient.astype('uint8')  # 將scale轉換成8-bit(簡報P55)
    return gradient, angles  # 回傳梯度以及角度

def non_maximum_suppression(img, angles):  # 非最大值響應，用以去除假的邊緣響應
    size = img.shape
    suppressed = np.zeros(size)
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            # 依梯度方向(法向量方向):水平、垂直、+-45度
            if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                compare_value = max(img[i, j - 1], img[i, j + 1])
            elif (22.5 <= angles[i, j] < 67.5):
                compare_value = max(img[i - 1, j - 1], img[i + 1, j + 1])
            elif (67.5 <= angles[i, j] < 112.5):
                compare_value = max(img[i - 1, j], img[i + 1, j])
            else:
                compare_value = max(img[i + 1, j - 1], img[i - 1, j + 1])
            # 對於某一點P，若他的梯度值沒有比梯度方向兩側點大則其梯度值設為0
            # 此處則為若該處為最大值，才將其填入新圖中
            if img[i, j] >= compare_value:
                suppressed[i, j] = img[i, j]
    suppressed = np.multiply(suppressed, 255.0 / suppressed.max())# 不知道這步驟要幹嘛的
    return suppressed

# 雙門檻值，大於high為強像素，小於low為弱像素，介於兩者之間其周圍4連通或8連通若有強項素其為邊緣
def double(img, low, high):  
    double_threshold = np.zeros(img.shape)
    size = img.shape
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            if(img[i, j] > high):
                double_threshold[i, j] = 1
            elif(img[i, j] <= high and img[i, j] >= low):
                if(np.max(img[i - 1 : i + 2, j - 1 : j + 2]) >= high):
                    double_threshold[i, j] = 0
                else:
                    double_threshold[i, j] = 0
            else:
                double_threshold[i, j] = 0
    return double_threshold

def double_threshold_hysteresis(img, low, high):  
    weak = 50
    strong = 255
    size = img.shape
    result = np.zeros(size)
    weak_x, weak_y = np.where((img > low) & (img <= high))  # np.where滿足條件輸出X，不滿足輸出Y
    strong_x, strong_y = np.where(img >= high)
    result[strong_x, strong_y] = strong
    result[weak_x, weak_y] = weak
    dx = np.array((-1, -1, 0, 1, 1, 1, 0, -1))
    dy = np.array((0, 1, 1, 1, 0, -1, -1, -1))
    
    while len(strong_x):
        x = strong_x[0]
        y = strong_y[0]
        strong_x = np.delete(strong_x, 0)
        strong_y = np.delete(strong_y, 0)
        for direction in range(len(dx)):
            new_x = x + dx[direction]
            new_y = y + dy[direction]
            if((new_x >= 0 & new_x < size[0] & new_y >= 0 & new_y < size[1]) and (result[new_x, new_y]  == weak)):
                result[new_x, new_y] = strong
                np.append(strong_x, new_x)
                np.append(strong_y, new_y)
    result[result != strong] = 0
    return result


def Canny(img, low, high):
    img, angles = SobelFilter(img)
    cv2.imshow('test_edgestest_edges', img)
    gradient = np.copy(img)
    img = non_maximum_suppression(img, angles)
    # img = double_threshold_hysteresis(img, low, high)
    return img, gradient



img, file_name, file_extension = read_img('E:/Program_File/PYTHON/數位影像處理作業/HW_3/Test_Img/03.jpg')
edges, gradient = Canny(img, 10, 30)

test_edges = double(edges, 10, 30)
cv2.imshow('test_edges', test_edges)
# cv2.imwrite('./Test_Img/' + file_name + '_canny_test_test_edges.jpg',edges)




cv2.imshow('Canny', edges)
# cv2.imwrite('./Test_Img/' + file_name + '_canny_test_.jpg',edges)
cv2.waitKey(0)
cv2.destroyAllWindows()






# Edge-detection---Canny-detector
# https://github.com/StefanPitur/Edge-detection---Canny-detector/blob/master/canny.py