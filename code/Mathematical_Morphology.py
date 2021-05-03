import numpy as np

def basic_morph_gray(im, er_dil, sizeI=4, sizeJ=4):
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
                        
            if er_dil==0:
                newValue = np.min(subImg)
            else:
                newValue = np.max(subImg)
                        
            img[i, j] = newValue
            
    return img



def erode(img, sizeI=4, sizeJ=4):
    return basic_morph_gray(img, 0, sizeI, sizeJ)


def dilate(img, sizeI=4, sizeJ=4):
    return basic_morph_gray(img, 1, sizeI, sizeJ)


def opening(img, sizeI=4, sizeJ=4):
    return dilate(erode(img, sizeI, sizeJ), sizeI, sizeJ)


def openingResidue(img, sizeI=4, sizeJ=4):
    return (img.astype(np.int32) - opening(img, sizeI, sizeJ).astype(np.int32))


def closing(img, sizeI=4, sizeJ=4):
    return erode(dilate(img, sizeI, sizeJ), sizeI, sizeJ)


def closingResidue(img, sizeI=4, sizeJ=4):
    return (img.astype(np.int32) - closing(img, sizeI, sizeJ).astype(np.int32))


