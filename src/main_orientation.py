
import pandas as pd
from sample import Sample
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np

loccam_orientation_images        = "./datasets/sample_orientation_images/loccam/"
loccam_orientation_images_output = "./output_files/orientation_images/loccam/"
navcam_orientation_images        = "./datasets/sample_orientation_images/navcam/"
navcam_orientation_images_output = "./output_files/orientation_images/navcam/"

def string_to_tuple(tuple_string):
    output_tuple = tuple(int(number) for number in tuple_string.replace('(', '').replace(')', '').replace('...', '').split(', '))

    return output_tuple


def cartesian_slope(pointA, pointB, image_height):
    cartesian_pointA = (pointA[0], image_height - pointA[1])
    cartesian_pointB = (pointB[0], image_height - pointB[1])

    num = (cartesian_pointB[1]-cartesian_pointA[1])
    div = (cartesian_pointB[0]-cartesian_pointA[0])
    
    if div == 0:
        result_cartesian_slope = 1.57  #90º
    else:
        result_cartesian_slope = num/div

    return result_cartesian_slope


def slope_to_degrees(slope):
    
    if(slope == 1.57):
        angle = 90
    else:
        angle   = math.degrees(math.atan(slope))
        angle   = round(angle,2)

        if(angle < 0):
            angle = 180 + angle
    
    return angle


def obtain_orientation_errors(input_folder, output_folder):

    df_orientation = pd.read_csv(input_folder + "orientation_list.csv")

    orientation_file_list = df_orientation['File_Name'].values.tolist()
    end_point1_list = df_orientation['end_point_1(x,y)'].values.tolist()
    end_point2_list = df_orientation['end_point_2(x,y)'].values.tolist()
    centroid_list = df_orientation['centroid(x,y)'].values.tolist()

    abs_error_list = []
    for i in range(0,len(orientation_file_list)):
        image_name = input_folder + orientation_file_list[i]
        img_file = cv2.imread(image_name)
        height, width, channels = img_file.shape
    
        input_bbox = ((0,0),(width-1,height-1))
        input_centroid = string_to_tuple(centroid_list[i])
        
        sample      = Sample(bbox = input_bbox, centroid = input_centroid)
        mask_sample = sample.binaryMaskedImage(image_name)
        result, object_pointA, object_pointB, global_orientation_image = sample.object2DOrientation(mask_sample)

        output_image_file = output_folder + "orientation_output_N" + str(i+1) +".png"
        if(result == 1):
            detected_slope = cartesian_slope(object_pointA, object_pointB, height)

            tagged_pointA  = string_to_tuple(end_point1_list[i])
            tagged_pointB  = string_to_tuple(end_point2_list[i])
            tagged_slope   = cartesian_slope(tagged_pointA, tagged_pointB, height)

            tagged_angle = slope_to_degrees(tagged_slope)
            detected_angle = slope_to_degrees(detected_slope)

            simp_error = abs(detected_angle - tagged_angle)
            sat_error  = abs(180 - simp_error)

            if(sat_error < simp_error):
                abs_error = sat_error
            else:
                abs_error = simp_error

            abs_error = round(abs_error,2)
            abs_error_list.append(abs_error)
            print("Image:" + orientation_file_list[i] + "  Original slope:" + str(tagged_angle) + "   Detected slope: " \
                +  str(detected_angle) + "   Abs. error: " + str(abs_error))
            cv2.imwrite(output_image_file, global_orientation_image)
            

    avg_error = sum(abs_error_list)/len(abs_error_list)
    percentage_estimated  = (len(abs_error_list)/len(orientation_file_list))*100

    sum_dev  = 0
    for i in range(0,len(abs_error_list)):
        sum_dev = sum_dev + ((abs_error_list[i] - avg_error) ** 2)
    
    standard_deviation = math.sqrt( sum_dev/len(abs_error_list) )
    
    print("Images detected: " + str(percentage_estimated) + "%")
    print("Average error: " + str(avg_error))
    print("Standard deviation of error: " + str(standard_deviation))


    return abs_error_list

def segmented_error(error_list):
    less5_list,  less10_list, less15_list, less20_list, less25_list = ([] for i in range(5))
    less30_list, less35_list, less40_list, less45_list ,less50_list = ([] for i in range(5))
    less55_list, less60_list, less65_list, less70_list ,less75_list = ([] for i in range(5))
    less80_list, less85_list, less90_list = ([] for i in range(3))

    nested_errors = []
    nested_errors.append(less5_list)
    nested_errors.append(less10_list)
    nested_errors.append(less15_list)
    nested_errors.append(less20_list)
    nested_errors.append(less25_list)
    nested_errors.append(less30_list)
    nested_errors.append(less35_list)
    nested_errors.append(less40_list)
    nested_errors.append(less45_list)
    nested_errors.append(less50_list)
    nested_errors.append(less55_list)
    nested_errors.append(less60_list)
    nested_errors.append(less65_list)
    nested_errors.append(less70_list)
    nested_errors.append(less75_list)
    nested_errors.append(less80_list)
    nested_errors.append(less85_list)
    nested_errors.append(less90_list)
    
    
    for i in range(0, len(error_list)):
        for error_index in range(0, len(nested_errors)):
            if error_list[i] < (error_index + 1) * 5:
                nested_errors[error_index].append(error_list[i])
                break

    graph_bar_len = []
    for error_index in range(0, len(nested_errors)):
        bar_per = (len(nested_errors[error_index])/len(error_list))*100
        graph_bar_len.append(bar_per)


    graph_bar_len.append(0) # last data

    return graph_bar_len


def plot_bars_chart(error_list_loccam, error_list_navcam):

    loccam_bar = segmented_error(error_list_loccam)
    navcam_bar = segmented_error(error_list_navcam)

    x_axis = ('0','5','10','15', '20', '25', '30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80', '85', '90')
    y_pos = np.arange(len(x_axis))
    loccam_seg  = plt.bar(y_pos , loccam_bar, align='edge', width=0.45, edgecolor = 'black', alpha=1 , label = 'LocCam (71 images)')
    navcam_seg  = plt.bar(y_pos + 0.45, navcam_bar, align='edge', width=0.45, edgecolor = 'black', alpha=1 , label = 'NavCam (298 images)')
    
    loccam_labels=[f'{x:.2f}%' for x in loccam_seg.datavalues]
    for i in range(0, len(loccam_labels)):
        if loccam_labels[i]=='0.00%':
            loccam_labels[i] =''
    plt.bar_label(loccam_seg, labels = loccam_labels, fontsize=9, rotation=90, padding = 5)


    navcam_labels=[f'{x:.2f}%' for x in navcam_seg.datavalues]
    for i in range(0, len(navcam_labels)):
        if navcam_labels[i]=='0.00%':
            navcam_labels[i] =''
    
    
    plt.bar_label(navcam_seg, labels = navcam_labels, fontsize=9, rotation=90, padding = 5)
    plt.xticks(y_pos, x_axis)
    plt.xlim(-0.5,18.5)
    plt.ylim(0,90)
    plt.legend()
    plt.ylabel('Estimated images over the total (%)')
    plt.xlabel('Absolute error of the orientation (degrees)')
    plt.title('Error of the estimated orientation vs. tagged image')
    figure = plt.gcf()  
    figure.set_size_inches(10, 8) 
    #plt.savefig('./output_files/orientation_images/Orientation_chart.png', dpi=600, bbox_inches='tight')



if __name__ == "__main__":

    abs_error_list_loccam = obtain_orientation_errors(loccam_orientation_images, loccam_orientation_images_output)
    abs_error_list_navcam = obtain_orientation_errors(navcam_orientation_images, navcam_orientation_images_output)
    plot_bars_chart(abs_error_list_loccam, abs_error_list_navcam)
