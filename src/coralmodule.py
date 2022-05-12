import sys
import cv2
import time
import numpy as np
import os

# Importing coral files directory
sys.path.append('./3rdparty_coral/coral_inference')
import coral_inference

DEBUG = False

class CoralModule():
    def __init__(self, model_file, anchors_file, classes_file, tensor_size = (416,416), threshold = 0.5):
        self.model_file     = model_file      # Tflite yolo model
        self.anchors_file   = anchors_file    # Yolo anchor file
        self.classes_file   = classes_file    # Classes file
        self.tensor_size    = tensor_size     # Image size required for input tensor
        self.threshold      = threshold       # Score threshold for detections

        # Init model and allocate tensors
        self.interpreter    = coral_inference.make_interpreter(self.model_file)
        self.interpreter.allocate_tensors()

    def image_inference(self, image_file, threshold = 0.5):
        anchors        = coral_inference.get_anchors(self.anchors_file)
        classes        = coral_inference.get_classes(self.classes_file)
        n_classes      = len(classes)
        self.threshold = threshold
        
        # Read the image and its size
        orig_image     = cv2.imread(image_file)
        image_name     = os.path.basename(image_file)
        orig_w, orig_h = orig_image.shape[0:2][::-1]

        # Establish padded image size and scale
        # Orig image is scaled to tensor input size
        # Using padding to keep original aspect ratio
        tensor_w, tensor_h = self.tensor_size
        w_scale = tensor_w/orig_w
        h_scale = tensor_h/orig_h
        
        # The same operation as 
        # padded_scale = min(w_scale,h_scale)
        if(w_scale<h_scale):
            padded_scale  = w_scale
            # Know if offset is applied to x
            flag_x_offset = True
        else:
            padded_scale  = h_scale
            flag_x_offset = False

        # Padded image size
        padded_w = int(orig_w*padded_scale)
        padded_h = int(orig_h*padded_scale)
        offset   = (max(padded_w,padded_h) - min(padded_w,padded_h))/2
        inverse_padded_scale = 1/padded_scale
        
        # Crop frame to network input shape
        padded_image = coral_inference.letterbox_image(orig_image.copy(), self.tensor_size)
        # Add batch dimension uint8
        padded_image = np.expand_dims(padded_image, 0).astype(np.uint8)

        # Run inference on square padded image, getting boxes
        start = time.time()
        padded_boxes, scores, pred_classes = coral_inference.inference(self.interpreter, padded_image, anchors, n_classes, self.threshold)
        inf_time = time.time() - start
        inference_time = round(inf_time*1000,2)

        scaled_boxes = []
        for topleft, botright in padded_boxes:
            if flag_x_offset == True:
                topleft_x = (int(topleft[0]))            * inverse_padded_scale
                topleft_y = (int(topleft[1])   - offset) * inverse_padded_scale
                botright_x = int(botright[0])            * inverse_padded_scale
                botright_y = int(botright[1]   - offset) * inverse_padded_scale
            else:
                topleft_x  = (int(topleft[0])  - offset) * inverse_padded_scale
                topleft_y  =  int(topleft[1])            * inverse_padded_scale
                botright_x = (int(botright[0]) - offset) * inverse_padded_scale
                botright_y =  int(botright[1])           * inverse_padded_scale
            scaled_topleft = (topleft_x,  topleft_y)
            scaled_botright= (botright_x, botright_y)
            scaled_boxes.append(((scaled_topleft[0], scaled_topleft[1]), (scaled_botright[0], scaled_botright[1])))

        ## Debug ##
        if (DEBUG):
            print("[Image: " + str(image_name) + "] Inference time " + str(inference_time) + " ms.")
            if len(scaled_boxes) == 0:
                print("[Image: " + str(image_name) + "] No detection.")
            else:
                print("[Image: " + str(image_name) + "] " + str(len(scaled_boxes)) + " detections.")
            
        
        return scaled_boxes, scores, pred_classes, inference_time
        
 
    def draw_boxes(self, image_file, boxes, scores, pred_classes):
        image   = cv2.imread(image_file)
        classes = coral_inference.get_classes(self.classes_file)
        image_prediction = coral_inference.draw_boxes(image.copy(), boxes, scores, pred_classes, classes)

        ## Debug ##
        if (DEBUG):
            cv2.imshow("Detected image", image_prediction)
            cv2.waitKey(0)

        return image_prediction

    def box_intersection(self, bbox1, bbox2):
        IoU = coral_inference.iou(bbox1, bbox2)

        return IoU