import cv2
import numpy as np
import pytesseract
import pandas as pd

pytesseract.pytesseract.tesseract_cmd="C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"

cascade= cv2.CascadeClassifier("numberplate_haarcade.xml")

def extract_num(img):
    image = cv2.imread(img)
    image = cv2.resize(image,None,fx = 0.5,fy = 0.5)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    nplate = cascade.detectMultiScale(gray,1.1,4)

    
    for(x,y,w,h) in nplate:
        wT,hT,cT=image.shape
        a,b=(int(0.02*wT),int(0.02*hT))
        plate = image[y+a:y+h-a,x+b:x+w-b,:]

        kernal=np.ones((1,1),np.uint8)
        plate = cv2.dilate(plate,kernal,iterations=1)
        plate = cv2.erode(plate,kernal,iterations=1)
        plate_gray = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
        
        (thresh,plate) = cv2.threshold(plate_gray,119,126,cv2.THRESH_BINARY)

        read = pytesseract.image_to_string(plate)
        read = list([val for val in read if val.isalpha() or val.isnumeric()])
        read = "".join(read)
        print('\n\nThe detected number from the licence plate is ' + read)
        #print(read)

        cv2.imshow("plate",plate)
    
        cv2.rectangle(image,(x,y),(x+w , y+h),(51,51,255),2)

    cv2.imshow('result',image)
    if cv2.waitKey(0) == 113:
         exit()
    cv2.destroyAllWindows()
    df = pd.read_csv('dataSet_Fleet.csv')

    f=0

    for i in range(len(df.AssetID)):
        if read==df.AssetID[i]:
            f=1
            details = df[(df['AssetID'] == read)]
            break

    if(f==0):
        print('\n\nIllegal\n\n')
    else:
        print('\n\nLegal\n\n')
        print(details)
        print('\n\n')

extract_num("skoda.jpg")