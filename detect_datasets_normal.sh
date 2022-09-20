#!/bin/bash

# To be able to call .sh from any directory
file_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$file_path"

# Flags to select type of weights and parameters
REAL_WEIGHT=1
SYNTHETIC_WEIGHT=0
SYNTHETIC_AND_REAL_WEIGHT=1

FIELD_TEST=2
DETECTION_THRESHOLD=0.75

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

# Detections with real weights saved in output_files/darknet_detections
if [ $REAL_WEIGHT -eq 1 ];
then
    while read test_image; do
        filename=$(basename -- "$test_image")
        
        # Remove extension
        filename="${filename%.*}"
        ./3rdparty_darknet/darknet detector test $data_file ./training_files/cfg/test-yolov3.cfg ./output_files/darknet_weights/real/yolov3_REAL_best.weights "$test_image" -thresh $DETECTION_THRESHOLD -dont_show; 
        mv predictions.jpg ./output_files/darknet_detections/"$filename"_normal_real_det.jpg; 
        echo $filename
    done < $valid_file
fi

# Detections with synthetic weights saved in output_files/darknet_detections
if [ $SYNTHETIC_WEIGHT -eq 1 ];
then
    while read test_image; do
        filename=$(basename -- "$test_image")
        
        # Remove extension
        filename="${filename%.*}"
        ./3rdparty_darknet/darknet detector test $data_file ./training_files/cfg/test-yolov3.cfg ./output_files/darknet_weights/synthetic/yolov3_SYNTHETIC_best.weights "$test_image" -thresh $DETECTION_THRESHOLD -dont_show; 
        mv predictions.jpg ./output_files/darknet_detections/"$filename"_normal_syn_det.jpg; 
        echo $filename
    done < $valid_file
fi

# Detection with pretrained weights saved in output_files/darknet_detections
if [ $SYNTHETIC_AND_REAL_WEIGHT -eq 1 ];
then
    while read test_image; do
        filename=$(basename -- "$test_image")
        
        # Remove extension
        filename="${filename%.*}"
        ./3rdparty_darknet/darknet detector test $data_file ./training_files/cfg/test-yolov3.cfg ./output_files/darknet_weights/both/yolov3_BOTH_best.weights "$test_image" -thresh $DETECTION_THRESHOLD -dont_show; 
        mv predictions.jpg ./output_files/darknet_detections/"$filename"_normal_both_det.jpg; 
        echo $filename
    done < $valid_file
fi


