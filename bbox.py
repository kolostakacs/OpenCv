import cv2
import numpy as np
import pytesseract
import os
import OCR_matching
import preprocess
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

img2 = cv2.imread('D:\\Asztal\\Melo\\preprocess_proba\\kep3.jpg')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
h,w,c = img2.shape
img2 = cv2.resize(img2, (w // 4, h // 4))

blur = cv2.GaussianBlur(img2, (7,7),0)
thresh = cv2.threshold(blur, 0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
dilate = cv2.dilate(thresh, kernal, iterations=1)

cv2.imshow('dil',dilate)

cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])

for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    #if h > 10 and w > 10:
    cv2.rectangle(img2, (x,y), (x+w,y+h), (36,255,12),2)

cv2.imshow('bbox', img2)
cv2.waitKey(0)

#imgShow = img2.copy()
#imgMask = np.zeros_like(imgShow)

