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

def Spectrum(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    r,c = gray.shape[:2]
    fp = np.zeros([r,c])
    for x in range(r):
        for y in range(c):
            fp[x,y] = np.power(-1,x+y)* gray[x,y]
    F = np.fft.fft2(fp)
    Fshift = np.fft.fftshift(F)
    mag = 20 * np.log(np.abs(Fshift)+1)
    mag = mag/mag.max()*255
    g = np.uint8(mag)
    plt.imshow(g,'gray')
    plt.show()
    
def WienerFilter(img,cutoff,K):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    r,c = gray.shape[:2]
    fp = np.zeros([r,c])
    for x in range(r):
        for y in range(c):
            fp[x,y] = np.power(-1,x+y)* gray[x,y]
    F = np.fft.fft2(fp)
    G = F.copy()
    
    for u in range(r):
        for v in range(c):
            dist = np.sqrt((u-r/2)*(u-r/2) + (v-c/2)*(v-c/2))
            H = np.exp(-(dist*dist)/(2*cutoff*cutoff))
            H = H/(H*H+K)
            G[u,v]*=H
    
    gp = np.fft.ifft2(G)
    gp2 = np.zeros([r,c])
    for x in range(r):
        for y in range(c):
            gp2[x,y] = np.round(np.power(-1,x+y)* np.real(gp[x,y]))
    g = np.uint8(np.clip(gp2,0,255))
    
    return g

def GaussBlur(img):
    gaussBlur_img = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
    return gaussBlur_img

def Sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    dst = cv2.filter2D(img, -1, kernel=kernel)
    sharpen_img = dst
    return sharpen_img


## Detector
def BlurDetector (img):
    ddepth = cv2.CV_64F #64F # cv2.CV_16S  # 
#     kernel_size = 9 # It is "ksize"
#     grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    grayImg = img
    blur_score = cv2.Laplacian(grayImg, ddepth).var() 
#     blur_score = blur_score/10**6 
    return blur_score

def BrightnessDetector (img):
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    rows = grayImg.shape[0]
    cols = grayImg.shape[1]
    bright_value = np.sum(grayImg)/(255 * rows* cols)
    return bright_value

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse)) 
    return psnr
