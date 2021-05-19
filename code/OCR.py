import Mathematical_Morphology as morpho
import MeanShift as MeanShift
import Niblack as niblack
import LabelingRegions as LabelingRegions
import LettersNumbersClassification as LetNumClassif

import numpy as np


def classifyImages(imgs):
    return LetNumClassif.ClassifyLettersNumbers(imgs)


def segmentRegions(im, labels):
    current_meanY = -1
    row = 1

    labeled_letters = []
    for region_number, quantity in labels:
        if(region_number!=0):
            if(quantity > 10):
                letter = np.where(im==region_number, 1, 0)
                
                s0 = letter.shape[0]
                s1 = letter.shape[1]
                
                indices = np.where(letter==1)

                y0 = indices[0][0] - 5
                y1 = indices[0][-1] + 5
                x0 = indices[1][0] - 5
                x1 = indices[1][-1] + 5
                
                if(y0 < 0):
                    y0 = 0
                
                if(y1 >= s0):
                    y1 = s0 - 1
                    
                if(x0 < 0):
                    x0 = 0
                    
                if(x1 >= s1):
                    x1 = s1 - 1
                
                letter = letter[y0:y1, x0:x1]
                
                # mean value x and y for sorting
                in_m0 = int(indices[0].shape[0] / 2)
                in_m1 = int(indices[1].shape[0] / 2)
                meanval0 = indices[0][in_m0]
                meanval1 = indices[1][in_m1]

                if (current_meanY == -1):
                    current_meanY = meanval0

                if (current_meanY < meanval0 - 5):
                    current_meanY = meanval0
                    row = row + 1

                labeled_letters.append([letter, row, meanval1])
   
    labeled_letters_sorted = sorted(labeled_letters, key=lambda v: (v[1], v[2]))
    
    return labeled_letters_sorted




"""
labelingRegions: Labels each region of a binary image

input:
    - img: Image to label the regions
    - connectivity: Connectivity of the regions to be used, C4 or C8.

output:
    - im_out: Image result where all pixels of the same region have the same value
    - labels: Values that are used in the im_out for each region and the quantity of pixels
              that are in that region.
"""
def labelingRegions(img, connectivity='C8'):
    if (connectivity == 'C8'):
        [im_out, eq] = LabelingRegions.LabelingRegionsC8(img)
    else:
        im_out = LabelingRegions.LabelingRegions(img)
        
    unique, counts = np.unique(im_out.reshape(-1), return_counts=True)
    labels = np.asarray((unique, counts)).T
        
    return [im_out, labels]




"""
light_correction_and_binarize: It corrects the light in the image and binarize it

input:
    - img: The image wanted
    - method: 0 for Opening Residue + Mean shift, 1 for Niblack

output:
    - img_bin: image binarized
"""
def light_correction_and_binarize(img, method=0):
    
    if(method==0):
        
        img_op_res = morpho.openingResidue(img, 20, 20)
        X = img_op_res.reshape(-1, 1)
        finalPoints = MeanShift.MeanShift(X, h=10)
        
        clusterColors = {}
        colores = []
        cm = list(map(str, finalPoints))
        quantities = {}

        for c in cm:
            if c in clusterColors:
                colores.append(clusterColors[c])
                quantities[c] = quantities[c] + 1
            else:
                newColor = np.random.default_rng().uniform(0,255,3).astype(np.uint8)
                clusterColors[c] = newColor
                quantities[c] = 1
                colores.append(clusterColors[c])


        print("Number of clusters generated: " + str(len(clusterColors)))
        
        maxQ = 0
        maxQVal = 0
        minQ = 10
        minQVal = []

        imagen_clusterizada = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint32)
        asignaciones2 = np.round(finalPoints.reshape(img.shape[0], img.shape[1]), 5)

        for k,v in clusterColors.items():
            imagen_clusterizada[asignaciones2==round(float(k[1:-1]), 5)] = v

            if quantities[k] > maxQ:
                maxQ = quantities[k]
                maxQVal = v

            if quantities[k] < minQ:
                minQVal.append(v)
                
        image_letters = np.where(imagen_clusterizada!=maxQVal, 1, 0)

        for mv in minQVal:
            image_letters = np.where(imagen_clusterizada!=mv, image_letters, 0)

        image_letters = image_letters[:, :, 0]
        
    
    else:
        
        image_letters = niblack.NiblackBinarization(img, 15, 15, 0.3)

    return image_letters




"""
correct_perspective: It returns the image without perspective if it had.
It also crops the image to the main region.
"""
def correct_perspective(img):
    return img




"""
OCR main function. If it found text in the image, it returns it as string.

input:
    - im: The image that want to recognize the text
    - light_method: Method used in light correction and binarization. 0 for Opening Residue + Mean shift, 1 for Niblack
    - labelingConnectivity: C8 for connectivity-eight and C4 for connectivity-four

output:
    - text: The text found in the image as string
"""
def OCR(im, light_method=0, labelingConnectivity='C8'):
    
    im = correct_perspective(im)
    
    im = light_correction_and_binarize(im, light_method)
    
    [im, labels] = labelingRegions(im, labelingConnectivity)
    
    letters = segmentRegions(im, labels)
    
    text = classifyImages(letters)
    
    return text




