from picamera.array import PiRGBArray
import RPi.GPIO as GPIO
from picamera import PiCamera
import time
import cv2
import numpy as np
import math
#this is import functions used to move the car 
from gpio_init import *

theta=0
minLineLength = 900
maxLineGap = 10
#the camera that we use is pi cam 
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640, 480))
#we use sleep so the camera can work just fine then we start the code 
time.sleep(0.1)
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
   image = frame.array
   #ؤonvert  frame to gray color   
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   blurred = cv2.GaussianBlur(gray, (5, 5), 0)
   #we use canny edge detection to find edges 
   edged = cv2.Canny(blurred, 85, 85)
   #find lines we can create from these edges 
   #cv2.HoughLinesP(image, rho, theta, threshold, lines[ minLineLength, maxLineGap]) 
   #minLineLength - Minimum length of line. Line segments shorter than this are rejected.
   #maxLineGap - Maximum allowed gap between line segments to treat them as single line 
   lines = cv2.HoughLinesP(edged,1,np.pi/180,10,minLineLength,maxLineGap)
   if(lines.all() !=None):
       for x in range(0, len(lines)):
           for x1,y1,x2,y2 in lines[x]:
               cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
               theta=theta+math.atan2((y2-y1),(x2-x1))

#we choose threshold to be 6 by try and error we compare it with theta 
   threshold=6
   #If a line has a positive slope ( m > 0) then the line is going to the right 
   #so we need to return the car to left  
   if(theta>threshold):
       left(40)
   #If a line has a positive slope ( m > 0) then the line is going to the left 
   #so we need to return the car to right  
   if(theta<-threshold):
       right(40)
   if(abs(theta)<threshold):
      forward(36)

   theta=0
   cv2.imshow("Frame",image)
   # key we choose it to be "q"
   key = cv2.waitKey(1) & 0xFF
   rawCapture.truncate(0)
   # when we press q at the keyboard it matches the key  
   if key == ord("q"):
   	# we we press q we call function stop it stops the motors 
       stop()
    #GPIO.cleanup() to clean all action to avoid the error which appear when we run again 
       GPIO.cleanup()
       break