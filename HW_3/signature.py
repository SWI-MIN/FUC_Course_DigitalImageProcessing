import os
import cv2 as cv2
import numpy as np

def read_img(img_path):     # 讀檔, input = 影像路徑, output = 圖像,檔名,副檔名
    img_filepath = os.path.splitext(img_path)[0]    # 拆分路徑 & 副檔名，0 為路徑
    img_fileextension = os.path.splitext(img_path)[1]  # 1 為副檔名
    img_filename = os.path.basename(img_filepath)     # 取出檔名不含副檔名
    img = cv2.imread(img_path)  # 讀檔(中文路徑會爆掉)
    # img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8)) # 讀檔(含有中文路徑)
    return img, img_filename, img_fileextension  

def combine_different_size_images(foreground, background, start_y, start_x):
    foreground=cv2.addWeighted(background[start_y:start_y+foreground.shape[0], start_x:start_x+foreground.shape[1]],1,foreground,1,0)
    background[start_y:start_y+foreground.shape[0], start_x:start_x+foreground.shape[1]] = foreground
    return background

img_myself, file_name_myself, file_extension_myself = read_img('E:/Program_File/PYTHON/Digital_Image_Processing_HomeWork/HW_3/Test_Img/04.jpg')
img_signature, file_name_signature, file_extension_signature = read_img('E:/Program_File/PYTHON/Digital_Image_Processing_HomeWork/HW_3/Test_Img/SWIMIN.png')

signature = combine_different_size_images(img_signature, img_myself, 350, 50)

cv2.imshow('composited image', signature)
cv2.waitKey(0)
