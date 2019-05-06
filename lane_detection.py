import picamera
from picamera.array import PiRGBArray
import numpy as np
import cv2
import time
import warnings
warnings.filterwarnings('error')

image_size=(320, 192)
camera = picamera.PiCamera()
camera.resolution = image_size
camera.framerate = 7 
camera.vflip = False
camera.hflip = False 
#camera.exposure_mode='off'
rawCapture = PiRGBArray(camera, size=image_size)

# allow the camera to warmup
time.sleep(0.1)

# class for lane detection
class Lines():
    def __init__(self):
        # were the lines detected at least once
        self.detected_first = False
        # were the lines detected in the last iteration?
        self.detected = False
        # average x values of the fitted lines
        self.bestxl = None
        self.bestyl = None
        self.bestxr = None
        self.bestyr = None
        # polynomial coefficients averaged over the last iterations
        self.best_fit_l = None
        self.best_fit_r = None
        #polynomial coefficients for the most recent fit
        self.current_fit_l = None
        self.current_fit_r = None
        # radius of curvature of the lines in meters
        self.left_curverad = None
        self.right_curverad = None
        #distance in meters of vehicle center from the line
        self.offset = None
        # x values for detected line pixels
        self.allxl = None
        self.allxr = None
        # y values for detected line pixels
        self.allyl = None
        self.allyr = None
        # camera calibration parameters
        self.cam_mtx = None
        self.cam_dst = None
        # camera distortion parameters
        self.M = None
        self.Minv = None
        # image shape
        self.im_shape = (None,None)
        # distance to look ahead in meters
        self.look_ahead = 10
        self.remove_pixels = 90
        # enlarge output image
        self.enlarge = 2.5
        # warning from numpy polyfit
        self.poly_warning = False

    # set camera calibration parameters
    def set_cam_calib_param(self, mtx, dst):
        self.cam_mtx = mtx
        self.cam_dst = dst

    # undistort image
    def undistort(self, img):
        return cv2.undistort(img, self.cam_mtx, self.cam_dst, None,self.cam_mtx)
        
    # get binary image based on color thresholding
    def color_thresh(self, img, thresh=(0, 255)):
        # convert to HSV color space and separate the V channel
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        s_channel = hsv[:,:,2]

        # threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
        return s_binary

    # get binary image based on sobel gradient thresholding
    def abs_sobel_thresh(self, sobel, thresh=(0, 255)):

        abs_sobel = np.absolute(sobel)

        max_s = np.max(abs_sobel)
        if max_s == 0:
            max_s=1

        scaled_sobel = np.uint8(255*abs_sobel/max_s)

        sbinary = np.zeros_like(scaled_sobel)
        sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return sbinary

    # get binary image based on sobel magnitude gradient thresholding
    def mag_thresh(self, sobelx, sobely, mag_thresh=(0, 255)):

        abs_sobel = np.sqrt(sobelx**2 + sobely**2)

        max_s = np.max(abs_sobel)
        if max_s == 0:
            max_s=1

        scaled_sobel = np.uint8(255*abs_sobel/max_s)

        sbinary = np.zeros_like(scaled_sobel)
        sbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

        return sbinary

    # get binary image based on directional gradient thresholding
    def dir_threshold(self, sobelx, sobely, thresh=(0, np.pi/2)):

        abs_sobelx = np.abs(sobelx)
        abs_sobely = np.abs(sobely)
        grad_sobel = np.arctan2(abs_sobely, abs_sobelx)

        sbinary = np.zeros_like(grad_sobel)
        sbinary[(grad_sobel >= thresh[0]) & (grad_sobel <= thresh[1])] = 1

        return sbinary

    # get binary combining various thresholding methods
    def binary_extraction(self,image, ksize=3):
        # undistort first
        image = self.undistort(image)

        color_bin = self.color_thresh(image,thresh=(90, 150))              # initial values 110, 255

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize)

        gradx = self.abs_sobel_thresh(sobelx, thresh=(100, 190))             # initial values 40, 160
        grady = self.abs_sobel_thresh(sobely, thresh=(100, 190))             # initial values 40, 160
        mag_binary = self.mag_thresh(sobelx, sobely, mag_thresh=(100, 190))  # initial values 40, 160

        combined = np.zeros_like(gradx)
        combined[(((gradx == 1) & (grady == 1)) | (mag_binary == 1)) | (color_bin==1) ] = 1
        

        return combined

    # transform perspective
    def trans_per(self, image):

        image = self.binary_extraction(image)
        #set self.binary_image = combined image to use it later
        self.binary_image = image
        #save the image size
        ysize = image.shape[0]
        xsize = image.shape[1]

        # define region of interest(we need to find (top left ,top right ,bottom left , bottom right) points to find the area that contain the lane lines. 
        left_bottom = (xsize/10, ysize)
        #left_bottom if the size is 400*600 400 is x axis so we will start from pixel 40 &600 
        #why we do this ? because the image resolution will decrease due to the camera range  


        #to find the distance to look ahead in meters 
        apex_l = (xsize/2 - 2600/(self.look_ahead**2),  ysize - self.look_ahead*275/30)
        #we define look_ahead as we want after we create an object in the main or as default =10
        #CenterXpixels = Widht/2==>(xsize/2)
        #
        apex_r = (xsize/2 + 2600/(self.look_ahead**2),  ysize - self.look_ahead*275/30)
        right_bottom = (xsize - xsize/10, ysize)
        #right_bottom if the size is 400*600 400 is x axis so we will end to the pixel (400-40) &600 from the left bottom to the end of the image

        # define vertices for perspective transformation
        src = np.array([[left_bottom], [apex_l], [apex_r], [right_bottom]], dtype=np.float32)
        dst = np.float32([[xsize/3,ysize],[xsize/4.5,0],[xsize-xsize/4.5,0],[xsize-xsize/3, ysize]])
        #cv2.getPerspectiveTransform ==>given the source vertices array and the dis. vertices array to find the new matrix that we will use it on the source 
        #image to get the bied eye 
        self.M = cv2.getPerspectiveTransform(src, dst)
        
        self.Minv = cv2.getPerspectiveTransform(dst, src)

        if len(image.shape) > 2:
        # cv2.warpPerspective==>to get the wraped image given :source image ,transformation matrix ,shape of the new image (wraped)         
            warped = cv2.warpPerspective(image, self.M, image.shape[-2:None:-1], flags=cv2.INTER_LINEAR)
        else:
            warped = cv2.warpPerspective(image, self.M, image.shape[-1:None:-1], flags=cv2.INTER_LINEAR)
        return warped

    # creat window mask for lane detecion
    def window_mask(self, width, height, img_ref, center,level):
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height), \
               max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
        return output

    # find widow centroids of left and right lane
    def find_window_centroids(self, warped, window_width, window_height, margin):

        window_centroids = [] # Store the (left,right) window centroid positions per level
        window = np.ones(window_width) # Create our window template that we will use for convolutions

        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template

        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)

        # Add what we found for the first layer
        window_centroids.append((l_center,r_center))

        # Go through each layer looking for max pixel locations
        for level in range(1,(int)(warped.shape[0]/window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width/2
            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center+offset-margin,0))
            r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            # Add what we found for that layer
            window_centroids.append((l_center,r_center))

        return window_centroids

    
    # Draw polynomials on the extracted left and right lane 
    def get_fit(self, image):
        # If the lanes are not detected in last iteration we will search for them
        if not self.detected:
            # initaizlize window settings to search for lanes
            window_width = 40
            window_height = 40 # will result in 9 windows vertically because image size is 720
            margin = 10 # how much to slide left and right for searching

            #This function will return left points and right points of windows 
            window_centroids = self.find_window_centroids(image, window_width, window_height, margin)
            # if we found any window centers
            if len(window_centroids) > 0:
                # points used to draw all the left and right windows
                l_points = np.zeros_like(image)
                r_points = np.zeros_like(image)
                # go through each level and draw the windows
                for level in range(0,len(window_centroids)):
                    # Window_mask is a function to draw window areas
                    l_mask = self.window_mask(window_width,window_height,image,window_centroids[level][0],level)
                    r_mask = self.window_mask(window_width,window_height,image,window_centroids[level][1],level)
                    # Add graphic points from window mask here to total pixels found
                    l_points[(image == 1) & (l_mask == 1) ] = 1
                    r_points[(image == 1) & (r_mask == 1) ] = 1

                # construct images of the results
                template_l = np.array(l_points*255,np.uint8) # add left window pixels
                template_r = np.array(r_points*255,np.uint8) # add right window pixels
                zero_channel = np.zeros_like(template_l) # create a zero color channel
                left_right = np.array(cv2.merge((template_l,zero_channel,template_r)),np.uint8) # make color image left and right lane

                # get points for polynomial fit
                self.allyl,self.allxl = l_points.nonzero()
                self.allyr,self.allxr = r_points.nonzero()


                # check if lanes are detected
                if (len(self.allxl)>0) & (len(self.allxr)>0):
                    try:
                        self.current_fit_l = np.polyfit(self.allyl,self.allxl, 2)
                        self.current_fit_r = np.polyfit(self.allyr,self.allxr, 2)
                        self.poly_warning = False
                    except np.RankWarning:
                        self.poly_warning = True
                        pass

                    # check if lanes are detected correctly
                    if self.check_fit():
                        self.detected = True

                        # if this is the first detection initialize the best values
                        if not self.detected_first:
                            self.best_fit_l = self.current_fit_l
                            self.best_fit_r = self.current_fit_r
                        # if not then average with new
                        else:
                            self.best_fit_l = self.best_fit_l*0.6 + self.current_fit_l * 0.4
                            self.best_fit_r = self.best_fit_r*0.6 + self.current_fit_r * 0.4

                        # assign new best values based on this iteration
                        self.detected_first = True
                        self.bestxl = self.allxl
                        self.bestyl = self.allyl
                        self.bestxr = self.allxr
                        self.bestyr = self.allyr
                        self.left_right = left_right

                    # set flag if lanes are not detected correctly
                    else:
                        self.detected = False

        # if lanes were detected in the last frame, search area for current frame
        else:
            non_zero_y, non_zero_x = image.nonzero()

            margin = 10 # search area margin
            left_lane_points_indx = ((non_zero_x > (self.best_fit_l[0]*(non_zero_y**2) + self.best_fit_l[1]*non_zero_y + self.best_fit_l[2] - margin)) & (non_zero_x < (self.best_fit_l[0] *(non_zero_y**2) + self.best_fit_l[1]*non_zero_y + self.best_fit_l[2] + margin)))
            right_lane_points_indx = ((non_zero_x > (self.best_fit_r[0]*(non_zero_y**2) + self.best_fit_r[1]*non_zero_y + self.best_fit_r[2] - margin)) & (non_zero_x < (self.best_fit_r[0]*(non_zero_y**2) + self.best_fit_r[1]*non_zero_y + self.best_fit_r[2] + margin)))

            # extracted lef lane pixels
            self.allxl= non_zero_x[left_lane_points_indx]
            self.allyl= non_zero_y[left_lane_points_indx]

            # extracted rightt lane pixels
            self.allxr= non_zero_x[right_lane_points_indx]
            self.allyr= non_zero_y[right_lane_points_indx]

            # if lines were found
            if (len(self.allxl)>0) & (len(self.allxr)>0):
                try:
                    self.current_fit_l = np.polyfit(self.allyl,self.allxl, 2)
                    self.current_fit_r = np.polyfit(self.allyr,self.allxr, 2)
                except np.RankWarning:
                    self.poly_warning = True
                    pass

                # check if lanes are detected correctly
                if self.check_fit():
                    # average out the best fit with new values
                    self.best_fit_l = self.best_fit_l*0.6 + self.current_fit_l * 0.4
                    self.best_fit_r = self.best_fit_r*0.6 + self.current_fit_r * 0.4

                    # assign new best values based on this iteration
                    self.bestxl = self.allxl
                    self.bestyl = self.allyl
                    self.bestxr = self.allxr
                    self.bestyr = self.allyr

                    # construct images of the results
                    template_l = np.copy(image).astype(np.uint8)
                    template_r = np.copy(image).astype(np.uint8)

                    template_l[non_zero_y[left_lane_points_indx],non_zero_x[left_lane_points_indx]] = 255 # add left window pixels
                    template_r[non_zero_y[right_lane_points_indx],non_zero_x[right_lane_points_indx]] = 255 # add right window pixels
                    zero_channel = np.zeros_like(template_l) # create a zero color channel
                    self.left_right = np.array(cv2.merge((template_l,zero_channel,template_r)),np.uint8) # make color image left and right lane

                # set flag if lanes are not detected correctly
                else:
                    self.detected = False

    # check if lanes are detected correctly
    def check_fit(self):
        # Generate x and y values of the fit
        ploty = np.linspace(0, self.im_shape[0]-1, self.im_shape[0])
        left_fitx = self.current_fit_l[0]*ploty**2 + self.current_fit_l[1]*ploty + self.current_fit_l[2]
        right_fitx = self.current_fit_r[0]*ploty**2 + self.current_fit_r[1]*ploty + self.current_fit_r[2]

        # find max, min and mean distance between the lanes
        max_dist  = np.amax(np.abs(right_fitx - left_fitx))
        min_dist  = np.amin(np.abs(right_fitx - left_fitx))
        mean_dist = np.mean(np.abs(right_fitx - left_fitx))
        # check if the lanes don't have a big deviation from the mean
        if (max_dist > 250) |  (np.abs(max_dist - mean_dist)> 100) | (np.abs(mean_dist - min_dist) > 100) | (mean_dist<50) | self.poly_warning:
            return False
        else:
            return True
