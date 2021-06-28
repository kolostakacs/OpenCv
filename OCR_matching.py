import cv2
import numpy as np
import pytesseract
import os
import preprocess

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

path = ('D:\\Asztal\\Melo\\ocr')
images = []
classNames = []

myList = os.listdir(path)
print('total classes', len(myList))

#Import images #### if we dont have images
for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}',0)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


orb = cv2.ORB_create()
def findDes(images):
    desList=[]
    for img in images:
        kp,des = orb.detectAndCompute(img,None)
        desList.append(des)
    return desList

desList = findDes(images)
print(len(desList))

def findID(img,desList, thres=0):
    kp2,des2 =orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher()
    matchList = []
    finalVal = -1
    try:
        for des in desList:
            matches = bf.knnMatch(des,des2, k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
    if len(matchList) !=0:
        if max(matchList) > thres:
            finalVal = matchList.index(max(matchList))
    return finalVal



img2 = cv2.imread('D:\\Asztal\\Melo\\test_ocr\\OCR_test1.png')
imgOriginal = img2.copy()
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
id = findID(img2,desList)
print(id)


