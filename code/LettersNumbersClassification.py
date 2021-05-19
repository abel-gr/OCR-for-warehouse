import numpy as np
from joblib import load
from PIL import Image

def getClassIDfromChar(character, CaseSensitive=False):
    
    code = ord(character)
    
    if(code > 64 and code < 91):
        code = code - 7
        
    elif(code > 96 and code < 123):
        if(CaseSensitive):
            code = code - 13
        else:
            code = code - 39
        
    code = code - 48
    
    if(code == 183):
        if(CaseSensitive):
            code = 62
        else:
            code = 36
    elif(code == 151):
        if(CaseSensitive):
            code = 63
        else:
            code = 36
    elif(code == 193):
        if(CaseSensitive):
            code = 64
        else:
            code = 37
    elif(code == 161):
        if(CaseSensitive):
            code = 65
        else:
            code = 37
    
    return code

def getCharFromClassID(code, CaseSensitive=False):
    
    
    if(CaseSensitive):
        
        if(code == 62):
            return 'ç'
        elif(code == 63):
            return 'Ç'
        elif(code == 64):
            return 'ñ'
        elif(code == 65):
            return 'Ñ'
        
    else:
        
        if(code==36):
            return 'Ç'
        elif(code==37):
            return 'Ñ'
        
    
    code = code + 48
    
    if(code > 57 and code < 84):
        code = code + 7
        
    elif(code > 83 and code < 110):
        if(CaseSensitive):
            code = code + 13
        else:
            code = code + 39
        
    return chr(code)


def ClassifyLettersNumbers(imgs):
    
    clf = load('MLP.joblib')
    
    ln = ''
    
    crow = 1
    
    for im, row, meanval1 in imgs:

        im_c = Image.fromarray(im)
        im_c = np.asarray(im_c.resize((32, 21)))
        im_c = np.where(im_c == 1, 0, 1)

        y_pred = clf.predict_proba(im_c.reshape(1, -1))

        clase = y_pred.argmax(axis=1)
        
        lett = getCharFromClassID(clase)
        
        if(row != crow):
            ln = ln + '\n'
            crow = row
        
        ln = ln + lett
        
        #print(lett, row, meanval1)
        
    return ln











    