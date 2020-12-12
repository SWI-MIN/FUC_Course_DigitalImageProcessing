from cv2 import cv2
import numpy as np
from scipy import signal

def nms(g,angle):
    h,w = g.shape
    canvas = np.zeros_like(g)
    for i in range(1,h - 1):
        for j in range(1,w - 1):
            # 找當前點的梯度方向上的另外兩點
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                max_ = max(g[i, j - 1], g[i, j + 1])
            elif (22.5 <= angle[i, j] < 67.5):
                max_ = max(g[i - 1, j - 1], g[i + 1, j + 1])
            elif (67.5 <= angle[i, j] < 112.5):
                max_ = max(g[i - 1, j], g[i + 1, j])
            elif (112.5 <= angle[i, j] < 157.5):
                max_ = max(g[i + 1, j - 1], g[i - 1, j + 1])
            
            if g[i, j] > max_:
                # 加果當前點是最大的就保留
                canvas[i, j] = g[i, j]
    
    return canvas

def combine(img,c):
    i,j = c
    # dx、dy 分別是周圍8個點的位置
    dx = [-1,-1,-1,0,0,1,1,1]
    dy = [-1,0,1,-1,1,-1,0,1]

    for x,y in zip(dx,dy):
        if img[i + x,j + y] == 1:
            # 如果有弱像素存在把那個點設成強像素、再移動到那個點、用遞廻不斷查看下一個點
            img[i + x,j + y] = 2
            combine(img,(i + x,j + y))

def connected(img,low_threshold,high_threshold):
    h,w = img.shape
    # 加果是強像素設成2、弱像素設1、其他設0
    for i in range(h):
        for j in range(w):
            if img[i,j] >= high_threshold:
                img[i,j] = 2
            elif high_threshold > img[i,j] >= low_threshold:
                img[i,j] = 1
            else:
                img[i,j] = 0

    for i in range(1,h - 1):
        for j in range(1,w - 1):
            if img[i,j] == 2:
                # 確認強像素周圍的點
                combine(img,(i,j))
    
    img = np.where(img == 2,255,0)
    return np.uint8(img)

def sobel(img):
    kernel_x = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
    kernel_y = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
    gx = signal.convolve2d(img,kernel_x,'same','symm')
    gy = signal.convolve2d(img,kernel_y,'same','symm')
    g = np.abs(gx) + np.abs(gy)

    # 算梯度方向
    angle = np.degrees(np.arctan2(gy,gx))
    # 小於 0 的加上 180 讓它轉半圈
    # 因為方向都在對角(差180)
    angle[angle < 0] += 180
    return g,angle

def resize(img,size = 150):
    h,w = img.shape
    canvas = np.zeros((size,size),np.uint8)
    ratio = min(size / h,size / w)
    new_h,new_w = int(ratio * h),int(ratio * w)
    img = cv2.resize(img,(new_w,new_h))
    canvas[(size - new_h) // 2:(size - new_h) // 2 + new_h,(size - new_w) // 2:(size - new_w) // 2 + new_w] = img
    return canvas

def paste_signature(img,signature):
    h,w = img.shape
    signature = resize(signature)
    img[h - 150:,w - 150:] = cv2.bitwise_or(img[h - 150:,w - 150:],signature)


img = cv2.imread('house1.jpg',0)
# 簽名
signature = cv2.imread('signature.png',0)
img = cv2.resize(img,(img.shape[1] // 8,img.shape[0] // 8))

# 模糊
blur_img = cv2.GaussianBlur(img,(3,3),0)

# 找梯度值
g,angle = sobel(blur_img)
# 最大值抑制
border = nms(g,angle)
# 找有和強像素相連的弱像素
canvas = connected(border,100,200)
# 貼簽名
paste_signature(canvas,signature)

# hough transform
lines = cv2.HoughLines(canvas,1,np.pi / 180,80)
lines = np.squeeze(lines)
black = np.zeros_like(img)
for rho,theta in lines: 
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a)) 
    cv2.line(black,(x1,y1),(x2,y2),(255,0,0),1)
paste_signature(black,signature)

# cv2.imwrite('myCanny.jpg',canvas)

cv2.imshow('canny',canvas)
cv2.imshow('line',black)

cv2.waitKey(0)
cv2.destroyAllWindows()