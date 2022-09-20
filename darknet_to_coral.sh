#!/bin/bash

# To be able to call .sh from any directory
file_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$file_path"

# Flags to select type of weights and parameters
REAL_WEIGHT=0
SYNTHETIC_WEIGHT=0
SYNTHETIC_AND_REAL_WEIGHT=1

./generate_data.sh

if [ $REAL_WEIGHT -eq 1 ];
then
    python3 ./3rdparty_coral/darknet_to_keras/darknet_to_keras.py ./training_files/cfg/yolov3-tiny-coral.cfg ./output_files/darknet_weights/real/yolov3_tiny_REAL_best.weights ./output_files/keras_conversion/yolov3_tiny_REAL_best.h5
    python3 ./3rdparty_coral/keras_to_tflite/quantization_to_tflite.py ./output_files/keras_conversion/yolov3_tiny_REAL_best.h5 ./output_files/quantized_tflite/yolov3_tiny_REAL_best.tflite
    edgetpu_compiler -o ./output_files/compiled_coral -s ./output_files/quantized_tflite/yolov3_tiny_REAL_best.tflite
fi

if [ $SYNTHETIC_WEIGHT -eq 1 ];
then
    python3 ./3rdparty_coral/darknet_to_keras/darknet_to_keras.py ./training_files/cfg/yolov3-tiny-coral.cfg ./output_files/darknet_weights/synthetic/yolov3_tiny_SYNTHETIC_best.weights ./output_files/keras_conversion/yolov3_tiny_SYNTHETIC_best.h5
    python3 ./3rdparty_coral/keras_to_tflite/quantization_to_tflite.py ./output_files/keras_conversion/yolov3_tiny_SYNTHETIC_best.h5 ./output_files/quantized_tflite/yolov3_tiny_SYNTHETIC_best.tflite
    edgetpu_compiler -o ./output_files/compiled_coral -s ./output_files/quantized_tflite/yolov3_tiny_SYNTHETIC_best.tflite
fi

if [ $SYNTHETIC_AND_REAL_WEIGHT -eq 1 ];
then
    python3 ./3rdparty_coral/darknet_to_keras/darknet_to_keras.py ./training_files/cfg/yolov3-tiny-coral.cfg ./output_files/darknet_weights/both/yolov3_tiny_BOTH_best.weights ./output_files/keras_conversion/yolov3_tiny_BOTH_best.h5
    python3 ./3rdparty_coral/keras_to_tflite/quantization_to_tflite.py ./output_files/keras_conversion/yolov3_tiny_BOTH_best.h5 ./output_files/quantized_tflite/yolov3_tiny_BOTH_best.tflite
    edgetpu_compiler -o ./output_files/compiled_coral -s ./output_files/quantized_tflite/yolov3_tiny_BOTH_best.tflite
fi
