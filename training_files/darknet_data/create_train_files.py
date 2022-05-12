from fnc_create_index import *

# Script for creating a txt with all the paths 
# of train images (labeled and no-labeled)

syn_traindir =  "./datasets/train_images/synthetic/train"
syn_validdir =  "./datasets/train_images/synthetic/valid"

real_traindir =  "./datasets/train_images/real/train"
real_validdir =  "./datasets/train_images/real/valid"

save_dir =  "./datasets/train_images/darknet_labels/"

# They are used as darknet valid.txt file in *.data
create_indexfile(syn_traindir, save_dir, "synthetic_train.txt")
create_indexfile(real_traindir, save_dir, "real_train.txt")

# They are used as darknet train.txt file in *.data
create_indexfile(syn_validdir, save_dir, "synthetic_valid.txt")
create_indexfile(real_validdir, save_dir, "real_valid.txt")