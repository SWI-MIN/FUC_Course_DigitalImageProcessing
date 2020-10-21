# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt

# a = np.random.randint(0,255,size=1000)
# print(a)


# histogram = np.zeros(256, dtype=np.uint8)

# print(np.sum(a.flat == 0))
# for k in range(len(histogram)):
#     histogram[k] = np.sum(a.flat == k)
# plt.bar(range(len(histogram)), histogram[range(len(histogram))], alpha=0.9, width = 1, lw=1)


# # plt.legend()
# # plt.legend(loc="upper left")
# # plt.bar(X, 256, alpha=0.9, width = 0.35, facecolor = 'lightskyblue', edgecolor = 'white', label='one', lw=1)

# # plt.plot(256,50,histogram)

# plt.show()


import os
import cv2
import numpy as np
img_path='E:/Program_File/PYTHON/數位影像處理作業/P01_flower.jpg'
img_filepath = os.path.splitext(img_path)[0]    # 拆分路徑 & 副檔名，0 為路徑
img_fileextension = os.path.splitext(img_path)[1]  # 1 為副檔名
img_filename = os.path.basename(img_filepath)     # 取出檔名不含副檔名

image = cv2.imread(img_filename+img_fileextension, 0)
image = cv2.imread('P01_flower.jpg', 0)
 
lut = np.zeros(256, dtype = image.dtype )#创建空的查找表
 
hist,bins = np.histogram(image.flatten(),256,[0,256]) 
cdf = hist.cumsum() #计算累积直方图
cdf_m = np.ma.masked_equal(cdf,0) #除去直方图中的0值
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())#等同于前面介绍的lut[i] = int(255.0 *p[i])公式
cdf = np.ma.filled(cdf_m,0).astype('uint8') #将掩模处理掉的元素补为0
 
#计算
result2 = cdf[image]
result = cv2.LUT(image, cdf)
 
cv2.imshow("OpenCVLUT", result)
cv2.imshow("NumPyLUT", result2)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(img_filename+'_gray'+img_fileextension,image)




