import os
import cv2 as cv2
import numpy as np

def read_img(img_path):     # 讀檔, input = 影像路徑, output = 圖像,檔名,副檔名
    img_filepath = os.path.splitext(img_path)[0]    # 拆分路徑 & 副檔名，0 為路徑
    img_fileextension = os.path.splitext(img_path)[1]  # 1 為副檔名
    img_filename = os.path.basename(img_filepath)     # 取出檔名不含副檔名
    # img = cv2.imread(img_path, -1)  # 讀檔(中文路徑會爆掉)
    img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1) # 讀檔(含有中文路徑)
    return img, img_filename, img_fileextension  

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
    cv2.imshow('composited image', background)               # show出來時上面的標題，show背景圖
    # cv2.imwrite('./Test_Img/' + file_name_myself + '_signature.jpg',background)

    cv2.waitKey()

# 讀檔  目前只能用png疊png，不能png疊jpg     
img_myself, file_name_myself, file_extension_myself = read_img('E:/Program_File/PYTHON/數位影像處理作業/HW_3/Test_Img/selfie.png')
img_signature, file_name_signature, file_extension_signature = read_img('E:/Program_File/PYTHON/數位影像處理作業/HW_3/Test_Img/SWIMIN.png')

combine_different_size_images(img_signature, img_myself, 350, 50)  # 呼叫上面def的副函式，第一個參數是前景，第二個為背景，第三四個為開始位置 Y*X (Y:直的，X:橫的)

