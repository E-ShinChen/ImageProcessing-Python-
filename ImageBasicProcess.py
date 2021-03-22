import cv2, math
import numpy as np 

# 

def Equalize_Hist(img):
    equalize_img = img.copy()
    if img.shape[2]==3:
        BImg = img[:,:,0]
        GImg = img[:,:,1]
        RImg = img[:,:,2]
        equalize_BImg = cv2.equalizeHist(BImg)
        equalize_GImg = cv2.equalizeHist(GImg)
        equalize_RImg = cv2.equalizeHist(RImg)        
        equalize_img[:,:,0] = equalize_BImg
        equalize_img[:,:,1] = equalize_GImg
        equalize_img[:,:,2] = equalize_RImg
    else:
        equalize_img = cv2.equalizeHist(img)
    return equalize_img

def Gamma_Trans(img):  
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mean = np.mean(grayImg)
    gamma = math.log10(0.5)/math.log10(mean/255) # 公式計算gamma
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 顏色值为整数
    return cv2.LUT(img, gamma_table)  # ,mean,gamma  # 顏色查表，根據強度均勻化設置門檻。


def Clahe_Hist(img):
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = img.copy()
    if img.shape[2]==3:
        BImg = img[:,:,0]
        GImg = img[:,:,1]
        RImg = img[:,:,2]
        clahe_BImg = clahe.apply(BImg)
        clahe_GImg = clahe.apply(GImg)
        clahe_RImg = clahe.apply(RImg)        
        clahe_img[:,:,0] = clahe_BImg
        clahe_img[:,:,1] = clahe_GImg
        clahe_img[:,:,2] = clahe_RImg
    else:
        clahe_img = cv2.clahe.apply(img)
    return clahe_img

def variance_of_laplacian(image):
	# blur = int(blur)
    # if blur is np.nan() or 0: print('Please input (image, blur level).\n Blur must be integer')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm

