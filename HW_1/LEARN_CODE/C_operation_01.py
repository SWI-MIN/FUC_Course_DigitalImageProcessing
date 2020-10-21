import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('EXAMPLE.jpg',cv2.IMREAD_GRAYSCALE)
# P01_flower   test_gray  EXAMPLE

histogram = np.zeros(256, dtype=np.uint32) # 存放讀到的pixel值
cdf = np.zeros(256, dtype=np.uint32) # 累積分布函數（cdf）
histogram_cdf = np.zeros(256, dtype=np.uint32) # 存放乘以CDF的pixel值

for k in range(len(histogram)):
    histogram[k] = np.sum(img.flat == k)
    cdf[k] = cdf[k-1] + histogram[k]  # 取的cdf累計"值"

for k in range(len(histogram_cdf)):                # 畫出來為累積直方圖   
    histogram_cdf[k] = (cdf[k]-np.min(cdf))/(np.max(cdf)-np.min(cdf))*255
    # 出現這個值機率
# 另一個圖片去承接就圖片*機率值的結果

# print(histogram[100])
# print(np.max(cdf),np.min(cdf))
# print(histogram_cdf[100])

# plt.bar(range(len(histogram)), histogram, alpha=0.9, width = 1, lw=1)
# plt.show()
plt.bar(range(len(histogram_cdf)), histogram_cdf, alpha=0.9, width = 1, lw=1)
# plt.hist(histogram_cdf, bins=256, range=[0, 255])
plt.show()



# plt.figure()
# plt.title('Gray Scale histogram')
# plt.xlabel('gray scale value')
# plt.ylabel('pixels')
# plt.plot(255,255,histogram)

# plt.show()


# 參考資料 : https://mlog.club/article/758415
