from sample import Sample
from cameradepth import CameraDepth
from coralmodule import CoralModule
import cv2

type_LOCCAM = 0
type_NAVCAM = 1

input_image_1 = "./src/example_files/Loccam_example1_image.png"
input_image_2 = "./src/example_files/Loccam_example2_image.png"
input_image_3 = "./src/example_files/Navcam_example3_image.png"
input_image_4 = "./src/example_files/Navcam_example4_image.png"

input_depth_1 = "./src/example_files/Loccam_example1_depth.txt"
input_depth_2 = "./src/example_files/Loccam_example2_depth.txt"

classes = "./datasets/field_tests_images/darknet_labels/sample.names"
anchors = "./training_files/cfg/tiny_yolo_anchors.txt"
model   = "./output_files/compiled_coral/yolov3_tiny_BOTH_best_edgetpu.tflite"

coral_module = CoralModule(model, anchors, classes)
boxes, scores, pred_classes, inference_time = coral_module.image_inference(input_image_1, threshold = 0.75)
sample      = Sample(bbox=boxes[0])
mask_sample = sample.binaryMaskedImage(input_image_1)
result, object_pointA, object_pointB, global_orientation_image = sample.object2DOrientation(mask_sample)

# We declare camera and load the depth file
camera = CameraDepth(type_LOCCAM, width = 1024, height = 768, physical_height= 0.4)
depth_matrix = camera.loadDepth(input_depth_1)
coordA = camera.obtain3DGlobalcoord(depth_matrix, object_pointA, camera_angle = 30)
coordB = camera.obtain3DGlobalcoord(depth_matrix, object_pointB, camera_angle = 30)
object_size = camera.calc3DDistance(coordA,coordB)

print("Coordinates point A: " + str(coordA) + " meters")
print("Coordinates point B: " + str(coordB) + " meters")
print("Size of object: " + str(object_size) + " meters")
camera.calcDEM(depth_matrix, resolution = 0.01, camera_angle = 30)