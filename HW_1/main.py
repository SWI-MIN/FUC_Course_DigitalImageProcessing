from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

def equalize(img,bin_):
    def cumsum(bin_):
        tmp = np.zeros_like(bin_,np.int32)
        tmp[0] = bin_[0]
        for i,val in enumerate(bin_[1:],1):
            tmp[i] = tmp[i - 1] + val
        return tmp        # CDF累加的值

    cdf = cumsum(bin_) / (img.shape[0] * img.shape[1])
    min_,max_ = np.min(cdf),np.max(cdf)
    canvas = np.zeros_like(img,np.uint8)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            canvas[row,col] = np.uint8(np.round((cdf[img[row,col]] - min_) / (max_ - min_) * 255))

    return canvas

def histogram(img):
    bin_ = np.zeros((256,),np.int32)
    img_1d = np.reshape(img,(-1))
    for val in img_1d:
        # 對應數值的位置+1
        bin_[val] += 1
    return bin_      # 直方圖

def gray():
    img = cv2.imread('EXAMPLE.jpg',0)
    bin_ = histogram(img)
    canvas = equalize(img,bin_)
    bin_ = histogram(canvas)
    plt.bar(np.arange(bin_.shape[0]),bin_,align = 'center',color = 'b')
    plt.show()
    # plt.savefig('gray_hist_eq.png')
    plt.close()
    # cv2.imshow('img',img)
    # cv2.imshow('canvas',canvas)
    # cv2.imwrite('gray_eq.jpg',canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def color():
    img = cv2.imread('P01_flower.jpg')
    b_img,g_img,r_img = cv2.split(img)
    
    b_bin = histogram(b_img)
    g_bin = histogram(g_img)
    r_bin = histogram(r_img)

    b_img_eq = equalize(b_img,b_bin)
    g_img_eq = equalize(g_img,g_bin)
    r_img_eq = equalize(r_img,r_bin)

    b_bin_eq = histogram(b_img_eq)
    g_bin_eq = histogram(g_img_eq)
    r_bin_eq = histogram(r_img_eq)
    
    plt.subplot(2,3,1)
    plt.bar(np.arange(b_bin.shape[0]),b_bin,align = 'center',color = 'b')
    plt.subplot(2,3,4)
    plt.bar(np.arange(b_bin_eq.shape[0]),b_bin_eq,align = 'center',color = 'b')
    plt.subplot(2,3,2)

    plt.bar(np.arange(g_bin.shape[0]),g_bin,align = 'center',color = 'g')
    plt.subplot(2,3,5)
    plt.bar(np.arange(g_bin_eq.shape[0]),g_bin_eq,align = 'center',color = 'g')
    plt.subplot(2,3,3)
    
    plt.bar(np.arange(r_bin.shape[0]),r_bin,align = 'center',color = 'r')
    plt.subplot(2,3,6)
    plt.bar(np.arange(r_bin_eq.shape[0]),r_bin_eq,align = 'center',color = 'r')
    plt.show()
    # plt.savefig('color_hist_eq.png')
    plt.close()

    tmp = np.zeros_like(img)
    tmp[...,0] = b_img
    cv2.imwrite('b_img.jpg',tmp)
    tmp = np.zeros_like(img)
    tmp[...,1] = g_img
    cv2.imwrite('g_img.jpg',tmp)
    tmp = np.zeros_like(img)
    tmp[...,2] = r_img
    cv2.imwrite('r_img.jpg',tmp)

    tmp = np.zeros_like(img)
    tmp[...,0] = b_img_eq
    cv2.imwrite('b_img_eq.jpg',tmp)
    tmp = np.zeros_like(img)
    tmp[...,1] = g_img_eq
    cv2.imwrite('g_img_eq.jpg',tmp)
    tmp = np.zeros_like(img)
    tmp[...,2] = r_img_eq
    cv2.imwrite('r_img_eq.jpg',tmp)

    canvas = cv2.merge([b_img_eq,g_img_eq,r_img_eq])
    cv2.imwrite('merge.jpg',canvas)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

gray()
# color()