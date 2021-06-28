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

#if id==0:
roi = [[(1824, 1459), (2302, 1549), 'text', 'Name'],
       [(1817, 1541), (2344, 1612), 'text', 'City'],
       [(1815, 1605), (2443, 1680), 'text', 'Adress'],
       [(2492, 2151), (2890, 2228), 'numeric', 'Id']]
#elif id ==1 :

#elif id == 2 :

#elif id ==3:


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
       #cv2.imwrite('D:\\Asztal\\Melo\\cropp\\'+str(x)+'.jpg', imgCrop)

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