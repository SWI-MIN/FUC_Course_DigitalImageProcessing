from numpy import *
import cv2

# eye(4)    # 對角線為1??

# x = array([1,2,3,4])
# y = array([5,6,7,8])
# print(x-1)
# print(x+y)
# print(x-y)
# print(x*y)
# print(x@y)
# print(matmul(x,y))
# print(dot(x,y))    # x@y , matmul(x,y) , dot(x,y)  三者等價
# print('\n+++++++++++++++++++++++++\n')

# a = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# b = array([[1, 2, 3, 4, 5],
#        [6, 7, 8, 9, 10]])

# print('mean : ', mean(a,axis=0))  # 沿指定軸計算算術平均值。
# print('std : ', std(a))     # 計算沿指定軸的標準偏差。
# print('max : ', max(a))     # 找出最大值
# print(a[a < 4])   # 找出小於4

# print('mean : ', mean(b))
# print('std : ', std(b))
# print('std,axis = 0 : ', std(b,axis=0))
# print('std,axis = 1 : ', std(b,axis=1))
# print(b[b > 6])
# print('sum,axis = 0 : ', b.sum(axis=0))
# print('sum,axis = 1 : ', b.sum(axis=1))

# img_nz_ui = zeros([500,500], dtype='uint8')  # uint8 無符號整數（0 to 255）
# img_no_ui = ones([500,500], dtype='uint8')
# cv2.imshow('test001',img_nz_ui)  # 全黑圖片
# cv2.imshow('test002',img_no_ui)  # 全黑圖片

# img_nz_f32 = zeros([500,500], dtype='float32')
# img_no_f32 = ones([500,500], dtype='float32')
# cv2.imshow('test003',img_nz_f32)  # 全黑圖片
# cv2.imshow('test004',img_no_f32)  # 全白圖片
# cv2.waitKey()


# size = int(1E6)  # 這段不知道怎麼運作QAQ
# timeit('[for xx in range(size) : xx ** 2]',number = 1)
# timeit('[arange(size) ** 2]',number = 1)


# em = empty(10)
# print(em)

# emx = array([[1, 2, 3, 4, 5, 6],
#        [6, 7, 8, 9, 10, 11]])
# emy = array([[ 1, 6],[ 2, 7],[ 3 , 8],[ 4 ,9],[ 5,10],[ 6,11]])
# print(emx.reshape(3,4))  # 將原先的陣列轉為幾列幾行 (列, 行)
# print(emx.T)     # 行列互換
# print(emy.T)
# print(emx[0])      # 第一列([1 2 3 4 5 6])
# print(emx[1,2:5])  # 第二列 第二到第五個元素包含頭不包含尾([ 8  9 10])

# 指定陣列內元素的使用方式稱為切片(SLICE)
#   :   冒號表示頭尾, 例如: [1:4] 表示編號1到編號3元素 [:] 則是全部
#   ,   表示隔開維度

# emz = array([[0, 1, 2, 3, 4], [5, 6, 7 ,8, 9], [10, 11, 12, 13, 14]])
# print(emz[1])       # 表示第2列全部 [5 6 7 8 9]
# print(emz[1:])      # 第2列後全部  [[ 5  6  7  8  9]  [10 11 12 13 14]]
# print(emz[1,2:])    # 第2列從第2個元素到結束  [7 8 9]
# print(emz[:2,2:])   # [[2 3 4]  [7 8 9]]
# print(emz[:,2:4])   # [[ 2  3]  [ 7  8]  [12 13]]
# emz[oo,xx]]       # oo = 哪幾列 , xx = 第幾個元素到第幾個元素





# img_nz_ui8_bw = zeros([500,500], dtype='uint8')  

# for i in range(5):
#     for j in range(5):
#         if((i+j) % 2 == 0):
            # img_nz_ui8_bw[i*100:i*100+100,j*100:j*100+100] = 255
# img_nz_ui8_bwc = ~img_nz_ui8_bw
# cv2.imshow('test005',img_nz_ui8_bw)  # img 圖片，nz Numpy Zero，ui8 uint8，bw black and white
# cv2.imshow('test006',img_nz_ui8_bwc) # img 圖片，nz Numpy Zero，ui8 uint8，bw black and white，c change
# cv2.waitKey()

# img_nz_ui8_bw_x = img_nz_ui8_bw.reshape(250,1000)
# cv2.imshow('test007',img_nz_ui8_bw_x) # img 圖片，nz Numpy Zero，ui8 uint8，bw black and white，c change
# RESHAPE是照著元素順序重新排序，所以RESHAPE沒辦法對影像作想像中的”改變形狀“
# 元素的總數目固定，所以也不能“改變大小”
# TRY TRY opencv函數
# cv2.imshow('test008',cv2.resize(img_nz_ui8_bw,(500,100))) # img 圖片，nz Numpy Zero，ui8 uint8，bw black and white，c change
# cv2.waitKey()



# 建立一個影象,1000×1000大小,資料型別無符號8位
img_01 = zeros((1000,1000,3), dtype = 'uint8')
cv2.line(img_01,(0,0),(1000,1000),(0,255,0),10)       # 畫線 ——> 設定起點和終點，顏色，線條寬度 ——> 綠色，3個畫素寬度
cv2.rectangle(img_01,(10,10),(200,200),(0,255,255),1)  # 矩形 ——> 設定左上頂點和右下頂點，顏色，線條寬度
cv2.circle(img_01,(150,150),60,(0,0,255),-1)           # 圓 ——> 指定圓心和半徑，-1表示填充
cv2.ellipse(img_01,(256,256),(100,50),0,0,180,(20,213,79),-1) # 橢圓 ——> 中心點位置，長軸和短軸的長度，橢圓沿逆時針選擇角度，橢圓沿順時針方向起始角度和結束角度線型，-1表示填充
cv2.imshow('test009',img_01) 
cv2.waitKey()




















