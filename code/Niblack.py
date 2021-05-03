import numpy as np

def NiblackThreshold(im, sizeI=4, sizeJ=4, k=1.0):
    img = im.copy()
    
    s1 = img.shape[0]
    s2 = img.shape[1]
    
    sizeI = int(sizeI/2)
    sizeJ = int(sizeJ/2)
    
    if(sizeI <= 0):
        sizeI = 1
        
    if(sizeJ <= 0):
        sizeJ = 1
    
    for i in range(0, s1):
        for j in range(0, s2):
            
            posI1 = i-sizeI
            posI2 = i+sizeI
            
            posJ1 = j-sizeJ
            posJ2 = j+sizeJ
            
            if(posI1 < 0):
                posI1 = 0
            
            if(posI2 >= s1):
                posI2 = s1 - 1
                
            if(posJ1 < 0):
                posJ1 = 0
            
            if(posJ2 >= s2):
                posJ2 = s2 - 1
            
            subImg = im[posI1:posI2, posJ1:posJ2]
                        
            t = np.mean(subImg) + k * np.std(subImg)
                        
            img[i, j] = t
            
    return img

def NiblackBinarization(im, sizeI=4, sizeJ=4, k=1.0):
    t = NiblackThreshold(im, sizeI, sizeJ, k)

    img = im.copy()
    img[img <= t] = 0
    img[img > t] = 1
    
    return img