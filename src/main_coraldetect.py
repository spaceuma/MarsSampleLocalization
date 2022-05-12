import argparse
import os
import cv2
from coralmodule import CoralModule
import pandas as pd

DEBUG = False

parser = argparse.ArgumentParser("Run YOLO on google coral")
parser.add_argument("--model", required=True, help="Model to load.")
parser.add_argument("--anchors", required=True, help="Anchors file.")
parser.add_argument("--classes", required=True, help="Classes (.names) file.")
parser.add_argument("-t", "--threshold", help="Detection threshold.", type=float, default=0.5)
parser.add_argument("--valid_file", help="Valid file with image directories")
parser.add_argument("--detect_save", help="Folder to save detection images")
parser.add_argument("--detect_mode", help="Make detection of images")
parser.add_argument("--metrics_mode", help="Create metrics")

args = parser.parse_args()

def create_metrics(valid_file, conf_threshold):
    scores_list, iou_list, inference_list = [], [], []
    true_positives, false_positives, false_negatives, double_false_positives = 0, 0, 0, 0
    precision, recall = 0, 0 
    
    # Open valid.txt
    v_file      = open(args.valid_file)
    v_file_cont = v_file.read()
    image_list  = v_file_cont.splitlines()

    if "test1" in valid_file:
        df_loccam_left  = pd.read_csv("./datasets/field_tests_images/test1/test1_loccam_left.csv")
        df_loccam_right = pd.read_csv("./datasets/field_tests_images/test1/test1_loccam_right.csv") 
        df_navcam_right = pd.read_csv("./datasets/field_tests_images/test1/test1_navcam_right.csv") 
        df_navcam_left  = pd.read_csv("./datasets/field_tests_images/test1/test1_navcam_left.csv") 
    elif "test2" in valid_file:
        df_loccam_left  = pd.read_csv("./datasets/field_tests_images/test2/test2_loccam_left.csv")
        df_loccam_right = pd.read_csv("./datasets/field_tests_images/test2/test2_loccam_right.csv") 
        df_navcam_right = pd.read_csv("./datasets/field_tests_images/test2/test2_navcam_right.csv") 
        df_navcam_left  = pd.read_csv("./datasets/field_tests_images/test2/test2_navcam_left.csv") 


    for i in range(0, len(image_list)):
        df_used = []
        if "loccam_left" in image_list[i]:
            df_used = df_loccam_left
        elif "loccam_right" in image_list[i]:
            df_used = df_loccam_right
        elif "navcam_right" in image_list[i]:
            df_used = df_navcam_right
        elif "navcam_left" in image_list[i]:
            df_used = df_navcam_left

        boxes, scores, pred_classes, inference_time = module_coral.image_inference(image_list[i], threshold = conf_threshold)
        image_name = os.path.basename(image_list[i])
        df_image   = df_used[df_used['image'].str.contains(image_name)]
        
        # Image is labeled and detected
        label_box = []

        if len(boxes) > 1:
            # There are two detections but only one sample
            double_false_positives = double_false_positives + 1

        elif not df_image.empty and len(boxes) == 1:

            label_ymin = df_image['ymin'].item()
            label_ymax = df_image['ymax'].item()
            label_xmin = df_image['xmin'].item()
            label_xmax = df_image['xmax'].item()
            label_box.append(((label_xmin, label_ymin), (label_xmax, label_ymax)))

            ## Debug ##
            if(DEBUG):
                module_coral.draw_boxes(image_list[i], label_box, scores, pred_classes)

            # IoU
            iou = module_coral.box_intersection(label_box[0], boxes[0])

            scores_list.append(scores[0])
            iou_list.append(iou)
            inference_list.append(inference_time)
            print("[Image: " + str(image_name) + "] IoU: " + str(round(iou, 2)))
            print("[Image: " + str(image_name) + "] inferenced.")

            true_positives = true_positives + 1
        
        elif not df_image.empty and len(boxes) == 0:
            false_negatives = false_negatives +1

        elif df_image.empty and len(boxes) == 1:
            false_positives = false_positives +1

    return scores_list, iou_list, inference_list, true_positives, false_negatives, false_positives, double_false_positives


def save_detections(valid_file, save_folder, model_file, conf_threshold):
    v_file = open(valid_file)
    v_file_cont = v_file.read()
    image_list = v_file_cont.splitlines()

    for i in range(0, len(image_list)):
        boxes, scores, pred_classes, inference_time = module_coral.image_inference(image_list[i], threshold = conf_threshold)
        detected_image = module_coral.draw_boxes(image_list[i], boxes, scores, pred_classes)
        image_name     = os.path.basename(image_list[i])
        basename       = image_name.split('.png')[0]

        if "BOTH" in model_file:
            destination = save_folder + basename + "_coral_BOTH_det.png" 
        elif "SYNTHETIC" in model_file:
            destination = save_folder + basename + "_coral_SYNTHETIC_det.png"
        elif "REAL" in model_file:
            destination = save_folder + basename + "_coral_REAL_det.png"

        print("[Image: " + str(image_name) + "] Saving folder: " + save_folder)
        cv2.imwrite(destination, detected_image)


if __name__ == "__main__":

    module_coral = CoralModule(args.model, args.anchors, args.classes, threshold = args.threshold)

    if args.metrics_mode:
        scores_list, iou_list, inference_list, true_positives, false_negatives, false_positives, double_false_positives = create_metrics(args.valid_file, args.threshold)
        avg_scores    = sum(scores_list)/len(scores_list)
        avg_iou       = sum(iou_list)/len(iou_list)
        avg_inference = sum(inference_list)/len(inference_list)
        precision = true_positives / (true_positives + false_positives)
        recall    = true_positives / (true_positives + false_negatives)
        F1_score  = 2 * (precision * recall) / (precision + recall)
        
        print("Conf_threshold: "            + str(args.threshold))
        print("TP: "                        + str(true_positives))
        print("FN: "                        + str(false_negatives))
        print("Simple FP: "                 + str(false_positives))
        print("Double FP: "                 + str(double_false_positives))
        print("Total FP (Simple+Double): "  + str(double_false_positives + false_positives))
        print("Precision: "                 + str(round(precision, 2)))
        print("Recall: "                    + str(round(recall, 2)))
        print("F1_score: "                  + str(round(F1_score, 2)))
        print("Avg scores: "                + str(round(avg_scores, 2)))
        print("Avg iou: "                   + str(round(avg_iou, 2)))
        print("Avg inference time: "        + str(round(avg_inference, 2)) + " ms." )

    if args.detect_mode:
        save_detections(args.valid_file, args.detect_save, args.model, args.threshold)