
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
#  perspective transform
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
