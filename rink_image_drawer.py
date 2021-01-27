# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:10:53 2021

@author: JeffDryden
"""

from ctypes import *
import math
import random
import os
import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np



height = 544
width = 1280
camDistance = 550
viewingAngle = 78 #degrees
camHeight = 150
heightViewAngle = 4.5 #degrees
slopeWidth = math.tan(viewingAngle * math.pi / 180)
slopeHeight = math.tan(heightViewAngle * math.pi / 180)

anchorClasses = [4,5,6,7,9,10]
playerClasses = [0,1,2,3]

classDict = {
    0: {"Name": "TOR", "Colour": (91, 32, 0)},
    1: {"Name": "LEAFG", "Colour": (255,255,0)},
    2: {"Name": "MTL", "Colour": (45, 30, 175)},
    3: {"Name": "HABSG", "Colour": (255, 0, 255)},
    4: {"Name": "EXCEL", "FixX": 1230, "FixY": 414, "ConstantW": 400, "ConstantH": 850, "Angle": 31.468016, "Width": 0.107031}, #H 3002.674665500596 W 642.3408245746804
    5: {"Name": "ROGERS", "FixX": 220, "FixY": 544, "ConstantW": 420, "ConstantH": 630, "Angle": -21.072726619950554, "Width": 0.123438}, #286.7520440271298 W  553.5512653394878 H
    6: {"Name": "PS5", "FixX": 980, "FixY": 544, "ConstantW": 400 , "ConstantH": 650, "Angle": 17.264518700666944, "Width": 0.126563}, # H 881.5480968376409 W 573.5386922469687
    7: {"Name": "ADIDAS", "FixX": 42, "FixY": 404, "ConstantW": 360, "ConstantH": 610, "Angle": -32.08090010, "Width": 0.082031}, #Got -1137.1887963080944, used same as above , 833.1206812362283 H
    8: {"Name": "SNLOGO", "FixX": 0, "FixY": 0, "ConstantW": 0, "ConstantH": 0, "Angle": 0},
    9: {"Name": "CNTIRE", "FixX": 560, "FixY": 444, "ConstantW": 350, "ConstantH": 500, "Angle": -4.601412235566412, "Width": 0.145312}, # W 361.8850215854191 H  975.3118659647237
    10: {"Name": "CCOLA", "FixX": 710, "FixY": 444, "ConstantW": 350, "ConstantH": 500, "Angle": 0, "Width": 0.215625}, # W 513.4675 H  936.170

}

labelDict = {
    "TOR":0,
    "LEAFG":1,
    "MTL":2,
    "HABSG":3,
    "EXCEL":4,
    "ROGERS":5,
    "PS5":6,
    "ADIDAS":7,
    "SNLOGO":8,
    "CNTIRE":9,
    "CCOLA":10,
}


def rink_drawer(detections, imgNum, out):
    
    canProceed = False

    anchorPoints = 0
    anchorDetections = []
    playerDetections = []
    img = cv2.imread('RinkModel.png',1)
    anchorWidth = 0


    for detection in detections:
        #if labelDict[detection[0]] == 8:
            
        if labelDict[detection[0]] in anchorClasses:
            if (detection[2][2]/416) > 0.80*classDict[labelDict[detection[0]]]['Width']:
                anchorDetections.append(detection)
                anchorPoints+=1
                anchorWidth = (detection[2][2]/416)/classDict[labelDict[detection[0]]]['Width']
                canProceed = True
                
        elif labelDict[detection[0]] in playerClasses: 
            playerDetections.append(detection)

    if canProceed:   
        
        midPoint = []
        midPointX = 0
        midPointY = 0
        anchorClassDetected = []
        
        for detection in anchorDetections:
            midPoint = _midpointDetector(labelDict[detection[0]],detection[2][0]/416,detection[2][1]/416,anchorWidth)
            midPointX+=midPoint[0]
            midPointY+=midPoint[1]
            img = cv2.circle(img, (int(classDict[labelDict[detection[0]]]['FixX']),height-int(classDict[labelDict[detection[0]]]['FixY'])), radius=10, color=(0,0,0), thickness=-1)


            #plt.scatter(classDict[labelDict[detection[0]]]['FixX'], height-classDict[labelDict[detection[0]]]['FixY'], s=15, c='k', marker='x')

        midPoint = [midPointX/anchorPoints,midPointY/anchorPoints]
        img = cv2.circle(img, (int(midPoint[0]),height-int(midPoint[1])), radius=10, color=(0,255,255), thickness=-1)
        viewSlopeNorth, viewSlopeSouth = _cameraHeightAngle(midPoint[0],midPoint[1])
        viewSlopeLeft, viewSlopeRight = _cameraWidthAngle(midPoint[0],midPoint[1])
        heightCamBottomX,heightCamBottomY,heightCamBottomSlope,heightCamTopX,heightCamTopY = _bottomCameraAngleShift(midPoint[0],midPoint[1],viewSlopeNorth, viewSlopeSouth)
        img = _drawLinesOnRink(viewSlopeLeft, viewSlopeRight, viewSlopeNorth, viewSlopeSouth, heightCamBottomX, heightCamBottomY, heightCamBottomSlope,
                      heightCamTopX, heightCamTopY, img)
        
        for detection in playerDetections:
            playerX = 0
            playerY = 0
            for anchor in anchorDetections:
                playerLoc = _playerLocation(labelDict[detection[0]],labelDict[anchor[0]],(detection[2][0])/416,detection[2][1]/416,anchor[2][0]/416,anchor[2][1]/416, anchorWidth)
                playerX += playerLoc[0]
                playerY += playerLoc[1]
            playerLoc = [playerX/anchorPoints,playerY/anchorPoints]
            
            img = cv2.circle(img, (int(playerLoc[0]),height-int(playerLoc[1])), radius=10, color=classDict[labelDict[detection[0]]]["Colour"], thickness=-1)





    return img

def _midpointDetector(anchorClass,anchorX,anchorY, anchorWidth):
    FixX = classDict[anchorClass]['FixX']
    FixY = classDict[anchorClass]['FixY']
    angle = classDict[anchorClass]['Angle']
    diffW = (0.5-anchorY)*math.sin(math.radians(angle))+(anchorX - 0.5)*math.sin(math.radians(90+angle))
    diffH = (0.5-anchorY)*math.cos(math.radians(angle))+(anchorX - 0.5)*math.cos(math.radians(90+angle))
    shiftW = (diffW * classDict[anchorClass]['ConstantW'])/anchorWidth
    shiftH = (diffH * classDict[anchorClass]['ConstantH'])/anchorWidth
    midpointX, midpointY = FixX-shiftW, FixY-shiftH
    return [midpointX, midpointY]

def _playerLocation(detClass,anchorClass,detX,detY,anchorX,anchorY, anchorWidth):

    angle = classDict[anchorClass]['Angle']
    
    diffW = (detY-anchorY)*math.sin(math.radians(angle))+(anchorX - detX)*math.sin(math.radians(90+angle))
    diffH = (detY-anchorY)*math.cos(math.radians(angle))+(anchorX - detX)*math.cos(math.radians(90+angle))
    shiftW = (diffW * classDict[anchorClass]['ConstantW'])/anchorWidth
    shiftH = (diffH * classDict[anchorClass]['ConstantH'])/anchorWidth
    playerX, playerY = classDict[anchorClass]['FixX']-shiftW, classDict[anchorClass]['FixY']-shiftH
    
    cameraAngleHeight = math.degrees(math.atan(camHeight/(math.sqrt((playerX-(width/2))**2+(playerY+camDistance)**2))))
    cameraAngleWidth = 90-(math.degrees(math.atan((abs(playerX)-(width/2))/(playerY+camDistance))))
    
    if cameraAngleWidth < 7.5:
        totalAdjustment = math.tan(math.radians(cameraAngleHeight))*0
    else:    
        totalAdjustment = math.tan(math.radians(cameraAngleHeight))*800
    
    adjustmentX = totalAdjustment*math.cos(math.radians(cameraAngleWidth))
    adjustmentY = totalAdjustment*math.sin(math.radians(cameraAngleWidth))
    
    playerX,playerY = playerX-adjustmentX, playerY-adjustmentY
    
    return [playerX, playerY]    

def _informationGather(anchorClass,anchorX,anchorY,FixX,FixY,MidX,MidY):
    
    angle = math.degrees(math.atan((FixX-(width/2))/(camDistance+FixY)))
    diffW = (0.5-anchorY)*math.sin(math.radians(angle))+(anchorX - 0.5)*math.sin(math.radians(90+angle))
    diffH = (0.5-anchorY)*math.cos(math.radians(angle))+(anchorX - 0.5)*math.cos(math.radians(90+angle))
    ConstantW = (FixX-MidX)/diffW
    ConstantH = (FixY-MidY)/diffH
    print("Class is: ",classDict[anchorClass]['Name'])
    print("FixX is: ",FixX)
    print("FixY is: ",FixY)
    print("Angle is: ",angle)    
    print("ConstantW is: ",ConstantW)
    print("ConstantH is: ",ConstantH)

def _cameraHeightAngle(currentPosX,currentPosY):
    
    cameraAngleHeight = math.degrees(math.atan(camHeight/(math.sqrt((currentPosX-(width/2))**2+(currentPosY+camDistance)**2))))
    cameraAngleHeight = -1*cameraAngleHeight
    viewAngleNorthArray = np.array((1,slopeHeight))
    viewAngleSouthArray = np.array((1,-1*slopeHeight))
    theta = np.radians(cameraAngleHeight)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    viewAngleNorthArray = np.matmul(R,viewAngleNorthArray)
    viewAngleSouthArray = np.matmul(R,viewAngleSouthArray)
    viewSlopeNorth = viewAngleNorthArray[1]/viewAngleNorthArray[0]
    viewSlopeSouth = viewAngleSouthArray[1]/viewAngleSouthArray[0]
    
    return viewSlopeNorth, viewSlopeSouth

def _cameraWidthAngle(currentPosX,currentPosY):
    
    cameraAngleWidth = 90-(math.degrees(math.atan((abs(currentPosX)-(width/2))/(currentPosY+camDistance))))
    theta = np.radians(cameraAngleWidth)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    viewAngleLeftArray = np.array((1,slopeWidth))
    viewAngleRightArray = np.array((1,-1*slopeWidth))
    viewAngleLeftArray = np.matmul(R,viewAngleLeftArray)
    viewAngleRightArray = np.matmul(R,viewAngleRightArray)
    viewSlopeLeft = -1*viewAngleLeftArray[1]/viewAngleLeftArray[0]
    viewSlopeRight = -1*viewAngleRightArray[1]/viewAngleRightArray[0]
    
    return viewSlopeLeft,viewSlopeRight
    
def _bottomCameraAngleShift(currentPosX, currentPosY,viewSlopeNorth,viewSlopeSouth):
    
    angle = -1 * (currentPosX/abs(currentPosX))*(math.degrees(math.atan((abs(currentPosX)-(width/2))/(currentPosY+(camDistance)))))

    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    
    heightCamBottomCoords = np.array((0,-1*(camHeight/viewSlopeSouth)))
    heightCamBottomSlope = np.array((1,0))

    heightCamBottomCoords = np.matmul(R,heightCamBottomCoords)
    heightCamBottomSlope = np.matmul(R,heightCamBottomSlope)
    heightCamBottomX = heightCamBottomCoords[0]
    heightCamBottomY = heightCamBottomCoords[1]
    heightCamBottomSlope = heightCamBottomSlope[1]/heightCamBottomSlope[0]

    heightCamTopCoords = np.array((0,-1*(camHeight/viewSlopeNorth)))
    heightCamTopCoords = np.matmul(R,heightCamBottomCoords)
    heightCamTopX = heightCamBottomCoords[0]
    heightCamTopY = heightCamBottomCoords[1]
    
    return heightCamBottomX,heightCamBottomY,heightCamBottomSlope,heightCamTopX,heightCamTopY


def _drawLinesOnRink(viewSlopeLeft, viewSlopeRight, viewSlopeNorth, viewSlopeSouth, heightCamBottomX, heightCamBottomY, heightCamBottomSlope,
                   heightCamTopX, heightCamTopY, img):
    

    start = int((width/2) + camDistance*viewSlopeLeft)
    end = int((width/2) + (camDistance+height)*viewSlopeLeft)
    cv2.line(img,  (end, 0), (start, height),  (0,0,0), 4)
    start = int((width/2) + camDistance*viewSlopeRight)
    end = int((width/2) + (camDistance+height)*viewSlopeRight)
    cv2.line(img,  (end, 0), (start, height),  (0,0,0), 4)

    topLine = height - (-1 * int((camHeight/viewSlopeNorth)+camDistance))
    start = int(height-(heightCamBottomY-550) + heightCamBottomSlope * (heightCamTopX+(width/2)))
    end = int(height-(heightCamBottomY-550) - heightCamBottomSlope * (1280-(heightCamTopX+(width/2))))

    cv2.line(img, (0,start), (width,end),(0,0,0), 4)


    start = int(height-(heightCamBottomY-550) + heightCamBottomSlope * (heightCamBottomX+(width/2)))
    end = int(height-(heightCamBottomY-550) - heightCamBottomSlope * (1280-(heightCamBottomX+(width/2))))
    bottomLine = height - (-1 * int((camHeight/viewSlopeSouth)+camDistance))
    cv2.line(img, (0,start), (width,end),(0,0,0), 4)
    
    return img