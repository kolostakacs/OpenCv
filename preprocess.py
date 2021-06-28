import cv2
import numpy as np
import pytesseract
import os
import OCR_matching
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


imgQ = cv2.imread('D:\\Asztal\\Melo\\kep\\kep1.jpg')
h,w,c = imgQ.shape
imgQ = cv2.resize(imgQ, (w // 1, h //1))
#per = 3
#orb = cv2.ORB_create(10000000)
per = 10
orb = cv2.ORB_create(5000)
kp1,des1 = orb.detectAndCompute(imgQ,None)



path = 'D:\\Asztal\\Melo\\preprocess_proba'
myPicList = os.listdir(path)
print(myPicList)
for j,y in enumerate(myPicList):
    img = cv2.imread(path +"/"+y)
    img = cv2.resize(img, (w // 1, h // 1))
    kp2,des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2,des1)
    matches.sort(key=lambda x: x.distance)
    good = matches[:int(len(matches)*(per/100))]
    imgMatch = cv2.drawMatches(img,kp2,imgQ,kp1,good[:100],None, flags=2)
    #cv2.imshow(y,imgMatch)
    #cv2.waitKey(0)


    scrPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    M, _ =cv2.findHomography(scrPoints,dstPoints,cv2.RANSAC,5.0)
    imgScan = cv2.warpPerspective(img,M,(w,h))
    #imgScan = cv2.resize(imgScan, (w // 1, h // 1))
    #cv2.imshow(y,imgScan)
    cv2.waitKey(0)

filename = 'savedImage.jpg'

cv2.imwrite('D:\\Asztal\\Melo\\preprocess_proba\\uj_kep.jpg', imgScan)
print("After saving image:")


print('Successfully saved')