import cv2
import numpy as np
import matplotlib.pyplot as plt
import utilis
 

curveList=[]
avgVal=10

def getLaneCurve(img, display=2):
    imgCopy= img.copy()
    imgResult= img.copy()
    imgThres= utilis.thresholding(img)

    hT, wT, c= img.shape
    points= utilis.valTrackbars()
    imgWrap= utilis.warpImg(imgThres, points, wT, hT)
    imgWrapPoints= utilis.drawPoints(imgCopy, points)

    middlePoint, imgHist = utilis.getHistogram(imgWrap, display= True, minPer=0.5, region=4)
    curveAveragePoint, imgHist= utilis.getHistogram(imgWrap, display=True, minPer=0.9 )
    curveRaw = curveAveragePoint - middlePoint

    curveList.append(curveRaw)
    if len(curveList)>avgVal:
        curveList.pop(0)
    curve = int(sum(curveList)/len(curveList))

    if display!=0:
        imgInWrap= utilis.wrapImg(imgWrap, points, wT, hT, inv=True)
        imgInWrap = cv2.cvtColor(imgInWrap, cv2.COLOR_GRAY2BGR)
        imgInWrap[0:hT//3, 0:wT]==0,0,0
        imgLaneColor=np.zeros_like(img)
        imgLaneColor[:]=0,255,0
        imgLaneColor=cv2.bitwise_and(imgInWrap, imgLaneColor)
        imgResult= cv2.addWeighted(imgResult, 1, imgLaneColor, 1,0)
        midY= 450
        cv2.putText(imgResult, str(curve), (wT//2-80,85), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,255),3)
        cv2.line(imgResult, (wT//2, midY),(wT//2+(curve*3),midY),(255,0,255),5)
        cv2.line(imgResult,((wT//2+(curve*3)),midY-25),(wT//2+(curve*3),midY+25),(0,255,0),5) 
        for x in range(-30, 30):
            w= wT//20
            cv2.line(imgResult, (w * x+ int(curve//50),midY-10),(w * x + int(curve//50), midY+10),(0,0,255),2)
        #fps=cv2.getTickFrequency()/(cv2.getTickCount()-timer);
        #cv2.putText(imgResult, 'FPS'+str(int(fps)),(20,40),cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 50,50),3);

    if display == 2:
        imgStacked = utilis.stackImages(0.7, ([img, imgWrapPoints, imgWrap],[imgHist, imgLaneColor, imgResult]))
        cv2.imshow('ImageStack', imgStacked)
    elif display == 1:
        cv2.imshow('Resutlt', imgResult)

    curve = curve/100
    if curve>1: curve ==1
    if curve<-1:curve == -1
    return curve


if __name__ == '__main__':
    cap = cv2.VideoCapture('vid1.mp4')
    intialTrackBarVals = [102, 80, 20, 214 ]
    utilis.initializeTrackbars(intialTrackBarVals)
    frameCounter = 0
    while True:
        frameCounter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
 
        success, img = cap.read()
        img = cv2.resize(img,(480,240))
        curve = getLaneCurve(img,display=2)
        print(curve)
        #cv2.imshow('Vid',img)
        cv2.waitKey(1)