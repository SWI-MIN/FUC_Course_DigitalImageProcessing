import numpy as np
import cv2
from matplotlib import pyplot as plt

def whole_hist(image):
	'''
	繪製整幅影象的直方圖
	'''
	plt.hist(image.ravel(), 256, [0, 256]) # numpy的ravel函式功能是將多維陣列降為一維陣列
	plt.show()

# plt.hist()參數設置 : 

# arr: 需要計算直方圖的一維數組；
# bins: 直方圖的柱數，可選項，默認為10；
# density: : 是否將得到的直方圖向量歸一化。默認為0；
# color：顏色序列，默認為None；
# facecolor: 直方圖顏色；
# edgecolor: 直方圖邊框顏色；
# alpha: 透明度；
# histtype: 直方圖類型，『bar』, 『barstacked』, 『step』, 『stepfilled』；

# 參考資料 : https://kknews.cc/zh-tw/code/3ngaz5a.html

def channel_hist(image):
	'''
	畫三通道影象的直方圖
	'''
	color = ('b', 'g', 'r')   #這裡畫筆顏色的值可以為大寫或小寫或只寫首字母或大小寫混合
	for i , color in enumerate(color):
		hist = cv2.calcHist([image], [i], None, [256], [0, 256])  #計算直方圖
		plt.plot(hist, color)
		plt.xlim([0, 256])
	plt.show()

# cv2.calcHist(影像, 通道, 遮罩, 區間數量, 數值範圍)

# 影像：影像的來源，其型別可以是 uint8 或 float32，變數必須放在中括號當中，例如：[img]。
# 通道：指定影像的通道（channel），同樣必須放在中括號當中。若為灰階影像，則通道就要指定為 [0]，若為彩色影像則可用 [0]、[1] 或 [2] 指定 藍色、綠色或紅色的通道。
# 遮罩：以遮罩指定要納入計算的圖形區域，若指定為 None 則會計算整張圖形的所有像素。
# 區間數量：指定直方圖分隔區間的數量（bins），也就是圖形畫出來要有幾條長方形。
# 數值範圍：指定要計算的像素值範圍，通常都是設為 [0,256]（計算所有的像素值）。




image = cv2.imread('P01_flower.jpg')
# cv2.imshow('image', image)
# cv2.waitKey(0)
whole_hist(image)
# channel_hist(image)



