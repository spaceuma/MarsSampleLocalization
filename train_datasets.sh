#!/bin/bash

# To be able to call .sh from any directory
file_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$file_path"

# Flags to select type of training
REAL_TRAIN=0
SYNTHETIC_TRAIN=0
SYNTHETIC_AND_REAL_TRAIN=1

./generate_data.sh

# Training only with real dataset, saving training chart and weights 
# into /output_files/darknet_weights/real
if [ $REAL_TRAIN -eq 1 ];
then
    ./3rdparty_darknet/darknet detector train training_files/darknet_data/real_training.data training_files/cfg/yolov3-tiny-coral.cfg -map
    mv ./chart.png ./output_files/darknet_weights/real/yolo_REAL.png
    mv ./training_files/generated_weights/yolov3-tiny-coral_best.weights ./output_files/darknet_weights/real/yolov3_tiny_REAL_best.weights
fi

# Training only with synthetic dataset, saving training chart and weights 
# into /output_files/darknet_weights/synthetic
if [ $SYNTHETIC_TRAIN -eq 1 ];
then
    ./3rdparty_darknet/darknet detector train training_files/darknet_data/synthetic_training.data training_files/cfg/yolov3-tiny-coral.cfg -map
    mv ./chart.png ./output_files/darknet_weights/synthetic/yolo_SYNTHETIC.png
    mv ./training_files/generated_weights/yolov3-tiny-coral_best.weights ./output_files/darknet_weights/synthetic/yolov3_tiny_SYNTHETIC_best.weights
fi

# Use weights from synthetic images as pretraining
# in the training with real images. Training chart and final weights saved 
# into /output_files/darknet_weights/both
if [ $SYNTHETIC_AND_REAL_TRAIN -eq 1 ];
then
    ./3rdparty_darknet/darknet partial training_files/cfg/yolov3-tiny-coral.cfg ./output_files/darknet_weights/synthetic/yolov3_tiny_SYNTHETIC_best.weights  ./output_files/darknet_weights/synthetic/yolov3_synthetic_pre.conv.11 11
    ./3rdparty_darknet/darknet detector train training_files/darknet_data/real_training.data  training_files/cfg/yolov3-tiny-coral.cfg  ./output_files/darknet_weights/synthetic/yolov3_synthetic_pre.conv.11 -map
    mv ./chart.png ./output_files/darknet_weights/both/yolo_BOTH.png
    mv ./training_files/generated_weights/yolov3-tiny-coral_best.weights ./output_files/darknet_weights/both/yolov3_tiny_BOTH_best.weights
fi
