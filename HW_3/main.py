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

    # 平方開根號與取絕對值效果差不多，為求運算效率，有時會用絕對值取得近似值
    # gradient = abs(G_x) + abs(G_y) 
    gradient = np.sqrt(np.square(G_x) + np.square(G_y))
    gradient = np.multiply(gradient, 255.0 / gradient.max())      # 255.0 / gradient.max()將gradient轉成0-1，在乘回gradient就會變成0-255
    gradient = gradient.astype('uint8')  # 將scale轉換成8-bit(簡報P55)
    
    return gradient, angles  # 回傳梯度以及角度
            
def non_maximum_suppression(img, angles):  # 非最大值響應，用以去除假的邊緣響應
    size = img.shape
    suppressed = np.zeros(size, dtype = 'uint8')
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
    return suppressed

# 雙門檻值，大於high為強像素，小於low為弱像素，介於兩者之間其周圍4連通或8連通若有強項素其為邊緣
def double_threshold_hysteresis(img, low, high):  
    double_threshold = np.zeros(img.shape, dtype = 'uint8')
    size = img.shape
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            if(img[i, j] > high):
                double_threshold[i, j] = 255
            elif(img[i, j] <= high and img[i, j] >= low):
                if(np.max(img[i - 1 : i + 2, j - 1 : j + 2]) >= high):
                    double_threshold[i, j] = 255
                else:
                    double_threshold[i, j] = 0
            else:
                double_threshold[i, j] = 0
    return double_threshold

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

def HoughLinesP(canny):  # 定義Hough轉換參數
    HoughLinesP = canny.copy()
    lines = cv2.HoughLinesP(canny, rho=1, theta=np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10)
    for line in [l[0] for l in lines]:  # 畫線
        leftx, boty, rightx, topy = line
        cv2.line(HoughLinesP, (leftx, boty), (rightx,topy), (255, 255, 0), 2)
    return HoughLinesP

# 目前只能用png疊png，不能png疊jpg  
def combine_different_size_images(image1, image2, start_y, start_x):   
    foreground, background = image1.copy(), image2.copy()  # 第一張圖是要覆蓋的，第二章式背景，image1複製給這個參數(foreground)，後面以此類推
    # Check border
    if foreground.shape[0]+start_y > background.shape[0] or foreground.shape[1]+start_x > background.shape[1]: 
        raise ValueError("The foreground image exceeds the background boundaries at this location")

    alpha =1
    # do composite at specified location
    end_y = start_y+foreground.shape[0]  # 結束位置就是開始加上要覆蓋的圖的長度( y )
    end_x = start_x+foreground.shape[1]
    # 合併兩張圖，圖片、權重、圖片、權重、加到每個總和的標量，相當於調亮度、OutputArray(有沒有他好像沒差)
    combine = cv2.addWeighted(foreground, alpha, background[start_y:end_y, start_x:end_x,:], alpha, 0, background)
    background[start_y:end_y, start_x:end_x,:] = combine       # 合併的圖片貼回背景圖
    
    return background

img, file_name, file_extension = read_img('E:/Program_File/PYTHON/數位影像處理作業/HW_3/Test_Img/03.jpg')

canny, gradient = Canny(img, 10, 30)
HoughLinesP = HoughLinesP(canny)


# 讀檔  目前只能用png疊png，不能png疊jpg     
# img_myself, file_name_myself, file_extension_myself = read_img('E:/Program_File/PYTHON/數位影像處理作業/HW_3/Test_Img/selfie.png')
# img_signature, file_name_signature, file_extension_signature = read_img('E:/Program_File/PYTHON/數位影像處理作業/HW_3/Test_Img/SWIMIN.png')
# signature = combine_different_size_images(img_signature, img_myself, 350, 50)  # 呼叫上面def的副函式，第一個參數是前景，第二個為背景，第三四個為開始位置 Y*X (Y:直的，X:橫的)






cv2.imshow('Canny', canny)
# cv2.imwrite('./Test_Img/' + file_name + '_canny.jpg',canny)
# cv2.imshow('HoughLinesP', HoughLinesP)
# cv2.imwrite('./Test_Img/' + file_name + '_HoughLinesP.jpg',HoughLinesP)
# cv2.imshow('composited image', signature)
# cv2.imwrite('./Test_Img/' + file_name_myself + '_signature.jpg',background)

cv2.waitKey(0)
cv2.destroyAllWindows()


