#!/bin/bash

# To be able to call .sh from any directory
file_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$file_path"

# Flags to select type of weights and parameters
REAL_TEST=1
SYNTHETIC_TEST=0
SYNTHETIC_AND_REAL_TEST=1
FIELD_TEST=1
CONF_THRESHOLD=0.75

./generate_data.sh

# We select the field test we want to test
if [ $FIELD_TEST -eq 1 ];
then
    valid_file=./datasets/field_tests_images/darknet_labels/field_test1_valid.txt
    
elif [ $FIELD_TEST -eq 2 ];
then
    valid_file=./datasets/field_tests_images/darknet_labels/field_test2_valid.txt
fi

# Logs files will be saved in the output_files/log folder
# Test with pre-trained weights
if [ $SYNTHETIC_AND_REAL_TEST -eq 1 ];
then
   python3 ./src/main_coraldetect.py  --detect_mode true -t ${CONF_THRESHOLD} --detect_save ./output_files/normal_coral_detection/ --valid_file $valid_file --classes ./datasets/field_tests_images/darknet_labels/sample.names --anchors training_files/cfg/yolov3_anchors.txt  --model ./output_files/compiled_coral/yolov3_BOTH_best_edgetpu.tflite
fi

# Test with real image weights
if [ $REAL_TEST -eq 1 ];
then
   python3 ./src/main_coraldetect.py  --detect_mode true -t ${CONF_THRESHOLD} --detect_save ./output_files/normal_coral_detection/ --valid_file $valid_file --classes ./datasets/field_tests_images/darknet_labels/sample.names --anchors training_files/cfg/yolov3_anchors.txt  --model ./output_files/compiled_coral/yolov3_REAL_best_edgetpu.tflite
fi

# Test with synthetic image weights
if [ $SYNTHETIC_TEST -eq 1 ];
then
   python3 ./src/main_coraldetect.py  --detect_mode true -t ${CONF_THRESHOLD} --detect_save ./output_files/normal_coral_detection/ --valid_file $valid_file --classes ./datasets/field_tests_images/darknet_labels/sample.names --anchors training_files/cfg/yolov3_anchors.txt  --model ./output_files/compiled_coral/yolov3_SYNTHETIC_best_edgetpu.tflite
fi
