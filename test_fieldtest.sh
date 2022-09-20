#!/bin/bash

# To be able to call .sh from any directory
file_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$file_path"

# Flags to select type of weights and parameters
REAL_TEST=0
SYNTHETIC_TEST=0
SYNTHETIC_AND_REAL_TEST=1
FIELD_TEST=2
DEM=0

./generate_data.sh

# We select the field test we want to test
if [ $FIELD_TEST -eq 1 ];
then
    valid_file=./datasets/field_tests_images/darknet_labels/field_test1_valid.txt
    
elif [ $FIELD_TEST -eq 2 ];
then
    valid_file=./datasets/field_tests_images/darknet_labels/field_test2_valid.txt
fi

# Test with pre-trained weights
if [ $SYNTHETIC_AND_REAL_TEST -eq 1 ];
then
   python3 ./src/main_fieldtest.py --valid_file $valid_file --DEM ${DEM} --save_folder "./output_files/fieldtest/" --classes ./datasets/field_tests_images/darknet_labels/sample.names --anchors training_files/cfg/tiny_yolo_anchors.txt  --model ./output_files/compiled_coral/yolov3_tiny_BOTH_best_edgetpu.tflite
fi

# Test with real image weights
if [ $REAL_TEST -eq 1 ];
then
   python3 ./src/main_fieldtest.py --valid_file $valid_file --DEM ${DEM} --save_folder "./output_files/fieldtest/" --classes ./datasets/field_tests_images/darknet_labels/sample.names --anchors training_files/cfg/tiny_yolo_anchors.txt  --model ./output_files/compiled_coral/yolov3_tiny_BOTH_best_edgetpu.tflite
fi

# Test with synthetic image weights
if [ $SYNTHETIC_TEST -eq 1 ];
then
   python3 ./src/main_fieldtest.py --valid_file $valid_file --DEM ${DEM} --save_folder "./output_files/fieldtest/" --classes ./datasets/field_tests_images/darknet_labels/sample.names --anchors training_files/cfg/tiny_yolo_anchors.txt  --model ./output_files/compiled_coral/yolov3_tiny_BOTH_best_edgetpu.tflite
fi

