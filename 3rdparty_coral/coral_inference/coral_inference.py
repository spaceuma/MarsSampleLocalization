import numpy as np
import cv2
from time import time
from coral_utils import *
import platform
import tflite_runtime.interpreter as tflite

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(model_path=model_file,experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,{'device': device[0]} if device else {})])

# Run YOLO inference on the image, returns detected boxes
def inference(interpreter, img, anchors, n_classes, threshold):
    input_details, output_details, net_input_shape = get_interpreter_details(interpreter)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run model
    interpreter.invoke()

    # Retrieve outputs of the network
    out1 = interpreter.get_tensor(output_details[0]['index'])
    out2 = interpreter.get_tensor(output_details[1]['index'])

    # Dequantize the outputs
    o1_scale, o1_zero = output_details[0]['quantization']
    out1 = (out1.astype(np.float32) - o1_zero) * o1_scale
    o2_scale, o2_zero = output_details[1]['quantization']
    out2 = (out2.astype(np.float32) - o2_zero) * o2_scale

    # Get boxes from outputs of network
    _boxes1, _scores1, _classes1 = featuresToBoxes(out1, anchors[[1, 2, 3]], 
            n_classes, net_input_shape, threshold)
    _boxes2, _scores2, _classes2 = featuresToBoxes(out2, anchors[[3, 4, 5]], 
            n_classes, net_input_shape, threshold)

    # This is needed to be able to append nicely when the output layers don't
    # return any boxes
    if _boxes1.shape[0] == 0:
        _boxes1 = np.empty([0, 2, 2])
        _scores1 = np.empty([0,])
        _classes1 = np.empty([0,])
    if _boxes2.shape[0] == 0:
        _boxes2 = np.empty([0, 2, 2])
        _scores2 = np.empty([0,])
        _classes2 = np.empty([0,])

    boxes = np.append(_boxes1, _boxes2, axis=0)
    scores = np.append(_scores1, _scores2, axis=0)
    classes = np.append(_classes1, _classes2, axis=0)
    if len(boxes) > 0:
        boxes, scores, classes = nms_boxes(boxes, scores, classes)

    return boxes, scores, classes

def draw_boxes(image, boxes, scores, classes, class_names):
    i = 0
    img = image.copy()
    colors = [0, 255, 0]

    for topleft, botright in boxes:
        # Detected class
        cl = int(classes[i])
        # Box coordinates
        topleft = (int(topleft[0]), int(topleft[1]))
        botright = (int(botright[0]), int(botright[1]))

        # Draw box and class
        cv2.rectangle(img, topleft, botright, colors, 2)
        textpos = (topleft[0]-2, topleft[1] - 3)
        score = scores[i] * 100
        cl_name = class_names[cl]
        text = str(cl_name) + str(round(score, 2))
        cv2.putText(img, text, textpos, cv2.FONT_HERSHEY_DUPLEX,
                0.45, colors, 1, cv2.LINE_AA)
        i += 1
    
    return img

def get_interpreter_details(interpreter):
    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]["shape"]

    return input_details, output_details, input_shape


