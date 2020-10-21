import numpy as np
import cv2
from matplotlib import pyplot as plt

def gray_hist(img_gray):
    '''
    plt.hist(x, bins=10, range=None, normed=False,   
    weights=None, cumulative=False, bottom=None,   
    histtype=u'bar', align=u'mid', orientation=u'vertical',   
    rwidth=None, log=False, color=None, label=None, stacked=False,   
    hold=None, **kwargs)
    
    x：此參數是數據序列。
    bins：此參數是可選參數，它包含整數，序列或字符串。
    range：此參數是可選參數，它是垃圾箱的上限和下限。
    density：此參數是可選參數，包含布爾值。
    weights：這個參數是一個可選參數，它是一個權重數組，形狀與x相同。
    bottom：此參數是每個容器底部基線的位置。
    histt​​ype：此參數是可選參數，用於繪製直方圖的類型。{'bar'，'barstacked'，'step'，'stepfilled'}
    align：此參數是可選參數，它控制如何繪製直方圖。{“左”，“中”，“右”}
    rwidth：此參數是可選參數，它是條形圖的相對寬度，是箱寬度的一部分
    log：此參數是可選參數，用於將直方圖軸設置為對數刻度
    color：此參數是可選參數，它是顏色規格或顏色規格序列，每個數據集一個。
    label：此參數是可選參數，它是一個字符串或匹配多個數據集的字符串序列。
    normed：此參數是可選參數，包含布爾值。它改用density關鍵字參數。
    '''
    plt.hist(img_gray.ravel(), bins=256, range=[0, 255])
    plt.show()


# image = cv2.imread('P01_flower.jpg')
img_gray = cv2.imread('P01_flower.jpg',cv2.IMREAD_GRAYSCALE)
# P01_flower   test_gray
# cv2.imshow('image', image)
# cv2.waitKey(0)
gray_hist(img_gray)

