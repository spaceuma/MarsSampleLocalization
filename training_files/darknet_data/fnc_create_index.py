import os
from natsort import natsorted

def create_indexfile(images_dir, save_dir, file_name):
    image_list = []
    for root, dirs, files in os.walk(images_dir):
        for name in files:
            if name.endswith((".png")):
                filepath = root + os.sep + name
                image_list.append(filepath)

    with open(save_dir + file_name, "w") as file:
        # Sort images numerically
        image_list = natsorted(image_list, key=lambda y: y.lower())
        for i in range(0,len(image_list)):
            file.write(image_list[i] + '\n') 