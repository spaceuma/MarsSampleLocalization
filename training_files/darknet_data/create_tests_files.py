from fnc_create_index import *

# Script for creating a txt with all the paths 
# of test images (labeled and no-labeled)

fieldtest1 =  "./datasets/field_tests_images/test1"
fieldtest2 =  "./datasets/field_tests_images/test2"

# Where valid.txt are saved
save_dir   =  "./datasets/field_tests_images/darknet_labels/"

# They are used as darknet valid.txt file in *.data
create_indexfile(fieldtest1, save_dir, "field_test1_valid.txt")
create_indexfile(fieldtest2, save_dir, "field_test2_valid.txt")