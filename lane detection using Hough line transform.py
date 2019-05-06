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
   #rho : The resolution of the parameter r in pixels. We use 1 pixel.
   #theta: The resolution of the parameter \theta in radians. We use 1 degree (CV_PI/180)
   #threshold: The minimum number of intersections to “detect” a line
   lines = cv2.HoughLinesP(edged,1,np.pi/180,10,minLineLength,maxLineGap)

   if(lines.all() !=None):
       for x in range(0, len(lines)):
           for x1,y1,x2,y2 in lines[x]:
            #lines: A vector that stores the parameters (x_{start}, y_{start}, x_{end}, y_{end}) of the detected lines
               cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
               #math.atan2((y2-y1),(x2-x1)) to find the slope of the line or theta 
               theta=theta+math.atan2((y2-y1),(x2-x1))
