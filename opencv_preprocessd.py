import cv2
import numpy as np
import pytesseract
import os
import OCR_matching
import preprocess
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

#img2 = cv2.imread('D:\\Asztal\\Melo\\preprocess_proba\\kep_iphone.jpg')
img2 = preprocess.imgScan

imgOriginal = img2.copy()
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
id = OCR_matching.findID(img2,OCR_matching.desList)
print(id)


roi =[[(448, 365), (578, 389), 'text', 'Name'],
      [(329, 562), (395, 589), 'numeric', 'Sum']]

    #roi = [[(1799, 1454), (2334, 1564), 'text', 'Name'],
           #[(1794, 1539), (2324, 1604), 'text', 'City'],
           #[(1804, 1604), (2419, 1684), 'text', 'Address'],
           #[(2494, 2169), (2844, 2254), 'numeric', 'ID'],
           #[(1259, 2249), (1584, 2354), 'numeric', 'Sum'],
           #[(1214, 2339), (1584, 2424), 'numeric', 'Deadline']]


imgShow = img2.copy()
#imgShow = cv2.bitwise_and(imgShow,imgShow)
imgMask = np.zeros_like(imgShow)

myData = []
print(f'Extracting Data from Form')

for x,r in enumerate(roi):

       cv2.rectangle(imgMask, (r[0][0],r[0][1]),(r[1][0],r[1][1]),(0,0,255),cv2.FILLED)
       imgShow = cv2.addWeighted(imgShow,0.99,imgMask,0.1,0)

       imgCrop = img2[r[0][1]:r[1][1],r[0][0]:r[1][0]]
       cv2.imshow(str(x), imgCrop)

       if r[2] == 'text':
              print(f'{r[3]} :{pytesseract.image_to_string(imgCrop)}')
              myData.append(pytesseract.image_to_string(imgCrop))
       if r[2] =='numeric':
              print(f'{r[3]} :{pytesseract.image_to_string(imgCrop)}')
              myData.append(pytesseract.image_to_string(imgCrop, lang='eng',config='--dpi 300 --psm 11 --oem 1 -c tessedit_char_whitelist=0123456789'))
with open('DataOutput.csv','a+') as f:
       for data in myData:
              f.write((str(data)+','))
       f.write('\n')



#'--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'
#cv2.imshow('output',imgShow)
cv2.waitKey(0)