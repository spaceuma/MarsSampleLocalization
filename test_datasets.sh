#!/bin/bash

# To be able to call .sh from any directory
file_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$file_path"

# Flags to select type of weights and parameters
REAL_TEST=1
SYNTHETIC_TEST=0
SYNTHETIC_AND_REAL_TEST=1
FIELD_TEST=2
IOU_THRESHOLD=0.5
CONF_THRESHOLD=0.75

./generate_data.sh

# We select the field test we want to test
if [ $FIELD_TEST -eq 1 ];
then
    valid_file=./datasets/field_tests_images/darknet_labels/field_test1_valid.txt
    data_file=./training_files/darknet_data/field_test1.data
elif [ $FIELD_TEST -eq 2 ];
then
    valid_file=./datasets/field_tests_images/darknet_labels/field_test2_valid.txt
    data_file=./training_files/darknet_data/field_test2.data
fi

# Logs files will be saved in the output_files/log folder
# Test with pre-trained weights
if [ $SYNTHETIC_AND_REAL_TEST -eq 1 ];
then
   ./3rdparty_darknet/darknet detector map $data_file ./training_files/cfg/test-yolov3-tiny-coral.cfg ./output_files/darknet_weights/both/yolov3_tiny_BOTH_best.weights -thresh ${CONF_THRESHOLD} -iou_thresh ${IOU_THRESHOLD} -points 0 > ./output_files/log/darknet_results_fieldtest${FIELD_TEST}_both.txt
fi

# Test with real image weights
if [ $REAL_TEST -eq 1 ];
then
   ./3rdparty_darknet/darknet detector map $data_file ./training_files/cfg/test-yolov3-tiny-coral.cfg ./output_files/darknet_weights/real/yolov3_tiny_REAL_best.weights -thresh ${CONF_THRESHOLD} -iou_thresh ${IOU_THRESHOLD} -points 0 > ./output_files/log/darknet_results_fieldtest${FIELD_TEST}_real.txt
fi

# Test with synthetic image weights
if [ $SYNTHETIC_TEST -eq 1 ];
then
   ./3rdparty_darknet/darknet detector map $data_file ./training_files/cfg/test-yolov3-tiny-coral.cfg ./output_files/darknet_weights/synthetic/yolov3_tiny_SYNTHETIC_best.weights -thresh ${CONF_THRESHOLD} -iou_thresh ${IOU_THRESHOLD} -points 0 > ./output_files/log/darknet_results_fieldtest${FIELD_TEST}_synthetic.txt
fi