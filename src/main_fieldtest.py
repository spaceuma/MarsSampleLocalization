from coralmodule import CoralModule
from sample import Sample
from cameradepth import CameraDepth
import re, argparse, datetime, os
from natsort import natsorted
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import cv2

type_LOCCAM = 0
type_NAVCAM = 1

DEBUG = False

parser = argparse.ArgumentParser("Run TF-Lite YOLO-V3 Tiny inference.")
parser.add_argument("--model", required=True, help="Model to load.")
parser.add_argument("--anchors", required=True, help="Anchors file.")
parser.add_argument("--classes", required=True, help="Classes (.names) file.")
parser.add_argument("-t", "--threshold", help="Detection threshold.", type=float, default=0.75)
parser.add_argument("--valid_file", help="Valid file with image directories")
parser.add_argument("--DEM", help="DEM generation")
parser.add_argument("--save_folder", help="Folder to save images")

args = parser.parse_args()

# Obtain depth files list given a directory
def obtain_depth_files(depth_files_dir):
    depth_list = []
    for root, dirs, files in os.walk(depth_files_dir):
        for name in files:
            if name.endswith((".txt")):
                filepath = root + os.sep + name
                depth_list.append(filepath)
    depth_list = natsorted(depth_list, key=lambda y: y.lower())
    
    return depth_list


# Find the depth file that correspond to the image
def find_match_depth(image_name, depth_file_list):
    selected_depth = 0

    # Extract image timestamp
    image_timestamp = re.search('_T(.*).png', image_name)
    image_timestamp = image_timestamp.group(1).split('-')
    image_time      = datetime.datetime(2022, 1, 1, 1, int(image_timestamp[0]), int(image_timestamp[1]), int(image_timestamp[2]))

    # Extract all depth files timestamps
    for i in range(0,len(depth_file_list)):
        depth_timestamp = re.search('_T(.*).txt', depth_file_list[i])
        depth_timestamp = depth_timestamp.group(1).split('-')
        depth_time = datetime.datetime(2022, 1, 1, 1, int(depth_timestamp[0]), int(depth_timestamp[1]), int(depth_timestamp[2]))

        # The first depth timestamp that is greater 
        #  or equal to the image timestamp is selected
        time_condition =(image_time.time() <= depth_time.time())
        if(time_condition):
            selected_depth = depth_file_list[i]
            break

    # No depth file found with greater time
    if(selected_depth == 0):
        # We select the last depth file
        selected_depth = depth_file_list[-1]


    return selected_depth



def calculate_fieldtest_data(coral_module, image_file, depth_file, DEM):
    image_DEM = None
    x_offset  = None
    
    boxes, scores, pred_classes, inference_time = coral_module.image_inference(image_file, threshold = 0.75)
    detected_image = coral_module.draw_boxes(image_file, boxes, scores, pred_classes)

    if len(boxes) > 0:
        sample      = Sample(bbox=boxes[0])
        mask_sample = sample.binaryMaskedImage(image_file)
        result, object_pointA, object_pointB, orientation_image = sample.object2DOrientation(mask_sample)

        if result==1:
            cv2.arrowedLine(detected_image, tuple(object_pointA), tuple(object_pointB),
                                            color = (0,0,255), thickness = 2)
            
            if "navcam" in image_file:
                camera = CameraDepth(type_NAVCAM, width = 1024, height = 768, physical_height= 1)
            elif "loccam" in image_file:
                camera = CameraDepth(type_LOCCAM, width = 1024, height = 768, physical_height= 0.4)

            # We declare camera and load the depth file
            depth_matrix = camera.loadDepth(depth_file)
            coordA = camera.obtain3DGlobalcoord(depth_matrix, object_pointA, camera_angle = 30)
            coordB = camera.obtain3DGlobalcoord(depth_matrix, object_pointB, camera_angle = 30)
            object_size = camera.calc3DDistance(coordA,coordB)

            ## Debug ##
            if(DEBUG):
                print("Coordinates point A: " + str(coordA) + " meters")
                print("Coordinates point B: " + str(coordB) + " meters")
                print("Size of object: " + str(object_size) + " meters")

            if(DEM == 1):
                image_DEM, x_offset = camera.calcDEM(depth_matrix, resolution=0.1, camera_angle=30)

    
    return detected_image, image_DEM, x_offset



if __name__ == "__main__":
    
    # Open valid.txt to obtain images
    v_file      = open(args.valid_file)
    v_file_cont = v_file.read()
    image_list  = v_file_cont.splitlines()

    coral_module = CoralModule(args.model, args.anchors, args.classes)

    if "test1" in args.valid_file:
        navcam_depth_dir = 'datasets/field_tests_images/test1/navcam_depth'
        navcam_depth_list = obtain_depth_files(navcam_depth_dir)
        loccam_depth_dir = 'datasets/field_tests_images/test1/loccam_depth'
        loccam_depth_list = obtain_depth_files(loccam_depth_dir)
    elif "test2" in args.valid_file:
        navcam_depth_dir = 'datasets/field_tests_images/test2/navcam_depth'
        navcam_depth_list = obtain_depth_files(navcam_depth_dir)
        loccam_depth_dir = 'datasets/field_tests_images/test2/loccam_depth'
        loccam_depth_list = obtain_depth_files(loccam_depth_dir)

    for i in range(0, len(image_list)):
        if "navcam" in image_list[i]:
            depth_list = navcam_depth_list
        elif "loccam" in image_list[i]:
            depth_list = loccam_depth_list

        selected_depth = find_match_depth(image_list[i], depth_list)
        detected_image, image_DEM, x_offset = calculate_fieldtest_data(coral_module, image_list[i], selected_depth, args.DEM)
        depth_name = os.path.basename(selected_depth)
        image_name = os.path.basename(image_list[i])

        if (args.save_folder is not None):
            save_folder = args.save_folder
            destination_detected_image = save_folder + str(image_name).replace(".png", "") + "_detection.png"
            print("[Image: " + str(image_name) + "] Saving folder: " + save_folder)
            cv2.imwrite(destination_detected_image, detected_image)

            if image_DEM is not None and args.DEM == 1:
                print(args.DEM)
                destination_DEM =save_folder + str(image_name).replace(".png", "") + "_DEM.png"
                print("[DEM Image: " + str(depth_name) + "] Saving folder: " + save_folder)
                fig = plt.figure()
                plt.imshow(image_DEM)
                plt.xlim(0,300)
                plt.ylim(x_offset+150,x_offset-150)
                plt.xlabel("y distance from camera (cm)")
                plt.ylabel("x distance + xoffset (cm)")
                cbar = plt.colorbar()
                cbar.ax.set_ylabel('Cm of elevation', rotation=270, labelpad=10)
                plt.savefig(destination_DEM)
                plt.close(fig)   

        ## Debug ##
        if(DEBUG):
            print("Image: " + image_name + " corresponds to depth file " + depth_name)


