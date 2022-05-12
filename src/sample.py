import math
import cv2
import numpy as np

DEBUG = False
CONTOUR_AREA_THRES = 100

class Sample():
    def __init__(self, bbox, centroid = None):

        # TO DO: Check that box coordinates are inside image
        # TO DO: Check that calculate centroid is inside image
        bbox = np.asarray(bbox)
        bbox[bbox < 0] = 0

        self.bbox              = bbox                         # ((x1,y1),(x2,y2))
        self.upper_left_point  = (int(self.bbox[0][0]), int(self.bbox[0][1])) # Rectangle pos1
        self.lower_right_point = (int(self.bbox[1][0]), int(self.bbox[1][1])) # Rectangle pos2
        self.image_offset      = self.upper_left_point        # Offset of cropped image
        if centroid == None:
            self.centroid      = self.calcCentroid()          # (x,y)
        else:
            self.centroid      = centroid
        self.real_position     = (0,0,0)                      # (x,y,z)
        self.real_orientation  = (0,0,0)                      # (Roll, Pitch, Yaw)
        
        ## Debug ##
        self.crop_img          =  0
        self.rect_img          =  0
        
    def calcCentroid(self):
        bbox_size     = np.subtract(self.lower_right_point, self.upper_left_point)
        bbox_center   = np.floor_divide(bbox_size, 2)
        bbox_centroid = np.add(self.image_offset, bbox_center)

        return bbox_centroid
    
    def imageIsGrayscale(self, image): 
        flag_gray = False
        
        b = image[:,:,0]
        g = image[:,:,1]
        r = image[:,:,2]
        
        if (b==g).all() and (b==r).all(): 
            flag_gray = True
        
        return flag_gray

    def calc2DDistance(self, v0, v1):
        euclidean_distance = math.sqrt((v0[0]-v1[0])**2 + (v0[1]-v1[1])**2)
        return euclidean_distance

    
    def binaryMaskedImage(self, image_path):
        # We read the image
        ori_img  = cv2.imread(image_path)

        # We crop the image with a rectangle given by bounding box detector
        crop_img = ori_img[self.upper_left_point[1]:self.lower_right_point[1],
                           self.upper_left_point[0]:self.lower_right_point[0]]
        
        # If image is not grayscale, we convert it
        if not self.imageIsGrayscale(crop_img):
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        
        # Convert a 3-channel image to uint8 1-channel image
        crop_img    = np.array(crop_img[:,:,0] ,dtype=np.uint8)

        # Apply a gaussian filter and a binary mask
        kernel_size = (3,3)
        crop_img    = cv2.GaussianBlur(crop_img, kernel_size, 0)
        
        # Apply sobel algorithm to find gardients created by sample tube
        x_sobel     = cv2.Sobel(crop_img,cv2.CV_64F,1,0)
        y_sobel     = cv2.Sobel(crop_img,cv2.CV_64F,0,1)
        abs_x_sobel = cv2.convertScaleAbs(x_sobel) 
        abs_y_sobel = cv2.convertScaleAbs(y_sobel)
        pro_img     = cv2.addWeighted(abs_x_sobel,0.5,abs_y_sobel,0.5,0)
        
        # Equalize histogram to enhance obtained gradients
        pro_img  = cv2.equalizeHist(pro_img)

        # Apply adaptive threshold to obtain bianry mask
        pro_img  = cv2.adaptiveThreshold(pro_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY , blockSize = 21, C = 0)
        
        # Morphological operations to remove spurious points
        pro_img = cv2.erode(pro_img, kernel_size,  iterations=3)
        pro_img = cv2.dilate(pro_img, kernel_size, iterations=3)


        rect_img = cv2.rectangle(ori_img,(self.upper_left_point[0], self.upper_left_point[1]),  
                                    (self.lower_right_point[0], self.lower_right_point[1]), 
                                    color = (0,255,0), thickness = 1)
        self.crop_img = crop_img
        self.rect_img = rect_img
        
        ## Debug ##
        if(DEBUG):
            cv2.imshow('Input image',rect_img)
            cv2.imshow('Processed image',pro_img)
            cv2.waitKey(1)

        return pro_img



    def object2DOrientation(self, local_binary_image):
        flag_orientation_obtained = 0

        # Search for contours
        contours, hierarchy = cv2.findContours(local_binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
        local_centroid      = np.subtract(self.centroid,self.image_offset)  
        
        # See if contours were found
        flag_found_contours = 0
        if len(contours) > 0:
            flag_found_contours = 1

        if(flag_found_contours):
            contours_list, distances_list = [], []
            for contour in contours:
                # We check if centroid is enclosed by contour
                flag_centroid_inside = cv2.pointPolygonTest(contour, (local_centroid[0],local_centroid[1]), False)

                #Calculate the contour moments and areas
                if(flag_centroid_inside):
                    M = cv2.moments(contour)
                    
                    # Calculate area
                    contour_area = M['m00']
                    
                    if(contour_area > CONTOUR_AREA_THRES):
                        # Calculate contour's centroid (x,y)
                        contour_centroid   = (int(np.floor_divide(M['m10'],M['m00'])), int(np.floor_divide(M['m01'],M['m00'])))
                        
                        # Calculate distence between yolo detected centroid and contour centroid
                        centroids_distance = self.calc2DDistance(contour_centroid,local_centroid)
                        contours_list.append(contour)
                        distances_list.append(centroids_distance)
            
            # Verify if at least a contour has been chosen
            if contours_list:
                # Find the calculated contour whose centroid is closer to the given one 
                min_centroid_distance = min(distances_list)
                min_distance_index    = distances_list.index(min_centroid_distance) 
                selected_contour      = contours_list[min_distance_index]
                
                # Create a binary mask with the pixels enclosed by the contour
                contour_mask = np.zeros_like(local_binary_image)
                masked_image = np.zeros_like(local_binary_image)
                cv2.drawContours(contour_mask, [selected_contour], 0, 255, thickness = cv2.FILLED)
                
                # Being the sample an elongated object, find a line that fits the contour points
                vx, vy, x0, y0 = cv2.fitLine(selected_contour, cv2.DIST_L2, 0, 1e-2, 1e-2)
                slope  = vy / vx

                # Once the unit vectors of the line are defined
                # we make it go through the centroid keeping the slope
                x0, y0          = (local_centroid[0], local_centroid[1])
                line_y_points   = lambda x: int(np.floor(slope*(x-x0) + y0))

                # Select the points of the line that are inside
                # the boundary mask and create a list of then
                x_max = local_binary_image.shape[1]
                y_max = local_binary_image.shape[0]
                x_in_points_list = []
                for x in range(0,x_max):
                    y = line_y_points(x)
                    
                    if y >= 0 and y < y_max:
                        if contour_mask[y][x] > 0:
                            x_in_points_list.append(x)

                # Search for the enclosed point of the line that are
                # the most to the left and to the right
                if len(x_in_points_list) > 0:
                    min_intersect_point = min(x_in_points_list)
                    max_intersect_point = max(x_in_points_list)
                    orientation_pointA  = (min_intersect_point,line_y_points(min_intersect_point))
                    orientation_pointB  = (max_intersect_point,line_y_points(max_intersect_point))


                    masked_image[contour_mask == 255] = self.crop_img[contour_mask == 255]
                    local_orientation_image = cv2.arrowedLine(self.crop_img, orientation_pointA, orientation_pointB, 
                                                        color = (255,255,255), thickness = 1)
                    ## Debug ##
                    if(DEBUG):
                        cv2.imshow('Masked image',masked_image)
                        cv2.imshow('Orientation arrow',local_orientation_image)
                        cv2.waitKey(0)

                    # Convert the local end points  points coordinates to global
                    global_orientation_pointA = np.add(orientation_pointA, self.image_offset)
                    global_orientation_pointB = np.add(orientation_pointB, self.image_offset)
                    global_orientation_image  = cv2.arrowedLine(self.rect_img, tuple(global_orientation_pointA), tuple(global_orientation_pointB),
                                                color = (0,0,255), thickness = 1)
                    flag_orientation_obtained = 1
            
        
        if(flag_orientation_obtained):
            return 1, global_orientation_pointA, global_orientation_pointB, global_orientation_image
        else:
            return 0,0,0, None