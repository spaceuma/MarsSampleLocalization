#!/bin/bash

# Creation of training valid.txt and train.txt
python3 ./training_files/darknet_data/create_train_files.py

# Creation of test valid.txt
python3 ./training_files/darknet_data/create_tests_files.py

# Creation of folders to save generated weights and darknet detections
mkdir ./output_files/darknet_weights/real
mkdir ./output_files/darknet_weights/synthetic
mkdir ./output_files/darknet_weights/both
mkdir ./output_files/darknet_detections/
mkdir ./output_files/log/
mkdir ./output_files/keras_conversion/
mkdir ./output_files/quantized_tflite/
mkdir ./output_files/compiled_coral/
mkdir ./output_files/coral_detection/
mkdir ./output_files/fieldtest/

# Creation of folder to save darknet's training backup weights (specified in *.data)
mkdir ./training_files/generated_weights/

# Compilation of Darknet (see Makefile flags)
make --directory ./3rdparty_darknet/
