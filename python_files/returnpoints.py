#%%
import numpy as np
import cv2 
import scipy
from skimage import measure
from numpy import linalg as ln
from PIL import Image
import scipy
from tqdm import tqdm
import glob
import csv 
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
# from scipy.signal import argrelextrema
import os 
#%%
def cross_search(im_arr, thr, image_type):
    
    im_shape = np.shape(im_arr) 
    if_break = False
    coordinates = []
    total_sum = 0
    
    pack1 = 0
    pack2 = 0
    pack3 = 0
    pack4 = 0

    if image_type == 'Grid':
        print('GRID')
        im_shape = np.shape(im_arr)
        for i in range(int(im_shape[0]/4)):
            for j in range(int(im_shape[1]/4)):
                if i > thr and j > thr :
                    iter = 0
                    total_sum = 0
                    while iter <= thr:
                        if iter == 0:
                            sum  = im_arr[i][j]
                        else:
                            sum  = im_arr[i][j + iter] + im_arr[i][j - iter] + im_arr[i + iter][j] + im_arr[i - iter][j]
                        total_sum += sum
                        iter +=1
                
                    
                    if total_sum > int(thr*3.5): 
                        # print('success1')
                        y_temp = i
                        x_temp = j 
                        pack = (x_temp, y_temp)
                        coordinates.append(pack)
                        if_break = True
                        break
            if if_break == True:  
                if_break = False
                break
    
        for i in range(int(im_shape[0]/3)):
            for j in reversed(range(int(im_shape[1]))):
                if i >= thr and im_shape[1] - j > thr  :
                    iter = 0
                    total_sum = 0
                    while iter <= thr:
                        if iter == 0:
                            sum  = im_arr[i][j]
                        else:
                            sum  = im_arr[i][j + iter] + im_arr[i][j - iter] + im_arr[i + iter][j] + im_arr[i - iter][j]
                        total_sum += sum
                        iter +=1

                    if total_sum == 1 + int(thr*4): 
                        # print('success2')
                        y_temp = i
                        x_temp = j 
                        pack = (x_temp, y_temp)
                        coordinates.append(pack)
                        if_break = True
                        break
            if if_break == True: 
                if_break = False
                break
            
        for i in reversed(range(int(im_shape[0]))):
            for j in range(int(im_shape[1]/3)):
                if im_shape[0] - i > thr  and j > thr : 
                    iter = 0
                    total_sum = 0
                    while iter <= thr:
                        if iter == 0:
                            sum  = im_arr[i][j]
                        else:
                            sum  = im_arr[i][j + iter] + im_arr[i][j - iter] + im_arr[i + iter][j] + im_arr[i - iter][j]
                        total_sum += sum
                        iter +=1

                    if total_sum == 1 + int(thr*4):
                        # print('success3')
                        y_temp = i
                        x_temp = j 
                        pack = (x_temp, y_temp)
                        coordinates.append(pack)
                        if_break = True
                        break
            if if_break == True: 
                if_break = False
                break
            
        for i in reversed(range(int(im_shape[0]))):
            for j in reversed(range(int(im_shape[1]))):
                if im_shape[0] - i > thr and im_shape[1] - j > thr:
                    iter = 0
                    total_sum = 0
                    while iter <= thr:
                        if iter == 0:
                            sum  = im_arr[i][j]
                        else:
                            sum  = im_arr[i][j + iter] + im_arr[i][j - iter] + im_arr[i + iter][j] + im_arr[i - iter][j]
                        total_sum += sum
                        iter +=1


                    if total_sum == 1 + thr*4: 
                        # print('success4')
                        y_temp = i
                        x_temp = j 
                        pack = (x_temp, y_temp)
                        coordinates.append(pack)
                        if_break = True
                        break
            if if_break == True: break


    elif image_type == 'Image':
        print('IMAGE')
        diff1 = 0
        diff2 = 0
        x_diff3 = 0
        diff4 = 0
        
        while diff1 < int(im_shape[1]/4) - 100:
            # print('diff1', diff1)
            for i in range(int(im_shape[0]/4)):
                for j in range(int(im_shape[1]/4 - diff1)):
                    if i > thr and j > thr :
                        iter = 0
                        total_sum = 0
                        while iter <= thr:
                            if iter == 0:
                                sum  = im_arr[i][j]
                            else:
                                sum  = im_arr[i][j + iter] + im_arr[i][j - iter] + im_arr[i + iter][j] + im_arr[i - iter][j]
                            total_sum += sum
                            iter +=1

                        if total_sum > int(thr*3.5) :   
                            if diff1 == 0:
                                # print('in1')      
                                y_min= i   
                                x_min= j  
                                # print('ymin1', y_min)
                                # print('xmin1', x_min)
                                
                                pack1 = (x_min, y_min)
                                
                                if_break = True
                                break
                            
                            else:
                                # print('inin1')
                                y_temp = i   
                                x_temp = j  # to move it to the middle 
                                # print('ytemp', y_temp)
                                # print('xtemp', x_temp)
                                if x_temp < x_min:
                                    # print('total_sum', total_sum)
                                    # print('success1')   
                                    x_min = x_temp
                                    y_min = y_temp
                                    pack1 = (x_min, y_min)
                                    
                                
                                if_break = True
                                break

                
                if if_break == True:  
                    if_break = False
                    break
            diff1 += 20
        if pack1 == None:
            pack1 = (0,0)

        while diff2 < int(im_shape[1]/4) - 100:        
            for i in range(int(im_shape[0]/4)):
                for j in reversed(range(im_shape[1])):
                    if i >= thr and im_shape[1] - j > thr and j > im_shape[1] - int(im_shape[1]/4) + diff2  :
                        iter = 0
                        total_sum = 0
                        while iter <= thr:
                            if iter == 0:
                                sum  = im_arr[i][j]
                            else:
                                sum  = im_arr[i][j + iter] + im_arr[i][j - iter] + im_arr[i + iter][j] + im_arr[i - iter][j]
                            total_sum += sum
                            iter +=1

                        if total_sum > int(thr*3.5): 
                            # print('in2')
                            if diff2 == 0:
                                y_min= i   
                                x_min= j  
                                
                                pack2 = (x_min, y_min)

                                if_break = True
                                break
                            
                            else:
                                y_temp = i  
                                x_temp = j  # to move it to the middle 
                                # print('xtemp', x_temp)
                                if x_temp > x_min:
                                    # print('success2')   
                                    x_min = x_temp
                                    y_min = y_temp
                                    pack2 = (x_min, y_min)
                                    
                                if_break = True
                                break

                if if_break == True: 
                    if_break = False
                    break
            diff2 += 20
        
        if pack2 == None:
            pack2 = (0,0)

        while x_diff3 < int(im_shape[1]/4) - 100:        
            for i in reversed(range(im_shape[0])):
                for j in range(int(im_shape[1]/4 - x_diff3)):
                    if im_shape[0] - i >  thr  and j > thr and i > im_shape[0] - int(im_shape[1]/4): 
                        iter = 0
                        total_sum = 0
                        while iter <= thr:
                            if iter == 0:
                                sum  = im_arr[i][j]
                            else:
                                sum  = im_arr[i][j + iter] + im_arr[i][j - iter] + im_arr[i + iter][j] + im_arr[i - iter][j]
                            total_sum += sum
                            iter +=1

                        if total_sum > 1 + int(thr*3.5):
                            # print('in3')
                            if x_diff3 == 0:
                                # y_min= i - 2  
                                # x_min= j + 2 
                                y_min= i 
                                x_min= j 

                                pack3 = (x_min, y_min)

                                if_break = True
                                break

                            else:
                                y_temp= i   
                                x_temp= j  # to move it to the middle 
                                if x_temp < x_min:
                                    # print('success3')   
                                    x_min = x_temp
                                    y_min = y_temp
                                    pack3 = (x_min, y_min)
                                    
                                if_break = True
                                break

                
                if if_break == True: 
                    if_break = False
                    break
            x_diff3 += 20

        if pack3 == None:
            pack3 = (0,0)

        while diff4 < int(im_shape[1]/4) - 100:   
            # print(im_shape[1] - int(im_shape[1]/4) + diff4)   
            # print(im_shape[0] - int(im_shape[0]/4))   
            for i in reversed(range(im_shape[0])):
                for j in reversed(range(im_shape[1])):
                    if im_shape[0] - i > thr and im_shape[1] - j > thr and j > im_shape[1] - int(im_shape[1]/4) + diff4 and i > im_shape[0] - int(im_shape[0]/4):
                        iter = 0
                        total_sum = 0
                        while iter <= thr:
                            if iter == 0:
                                sum  = im_arr[i][j]
                            else:
                                sum  = im_arr[i][j + iter] + im_arr[i][j - iter] + im_arr[i + iter][j] + im_arr[i - iter][j]
                            total_sum += sum
                            iter +=1


                        if total_sum > int(thr*3.5): 
                            # print('total_sum', total_sum)
                            if diff4 == 0:
                                # print('in4')
                                # y_min= i - 2  
                                # x_min= j - 2
                                y_min= i   
                                x_min= j 
                                # print('ymin4', y_min) 
                                # print('xmin4', x_min) 

                                pack4 = (x_min, y_min)

                                if_break = True
                                break
                            
                            else:
                                y_temp = i - 2  
                                x_temp = j - 2 # to move it to the middle 
                                # print('ytemp', y_temp)
                                # print('xtemp', x_temp)
                                if x_temp > x_min:
                                    # print('success4')   
                                    x_min = x_temp
                                    y_min = y_temp
                                    pack4 = (x_min, y_min)
                        
                                if_break = True
                                break

                if if_break == True: 
                    if_break = False
                    break
            diff4 += 20
        if pack4 == None:
            pack4 = (0,0)
        coordinates.extend([pack1, pack2, pack3, pack4])
    
    if pack1 == (0,0):
        lower_width = ln.norm(coordinates[2][0] - coordinates[3][0])
        right_height = ln.norm(coordinates[1][1] - coordinates[3][1])

    elif pack2 == (0,0):
        left_height = ln.norm(coordinates[0][1] - coordinates[2][1])
        lower_width = ln.norm(coordinates[2][0] - coordinates[3][0])
    
    elif pack3 == (0,0):
        upper_width = ln.norm(coordinates[0][0] - coordinates[1][0])
        right_height = ln.norm(coordinates[1][1] - coordinates[3][1])

    elif pack4 == (0,0):
        upper_width = ln.norm(coordinates[0][0] - coordinates[1][0])
        left_height = ln.norm(coordinates[0][1] - coordinates[2][1])

    else:
        upper_width = ln.norm(coordinates[0][0] - coordinates[1][0])
        left_height = ln.norm(coordinates[0][1] - coordinates[2][1])

        lower_width = ln.norm(coordinates[2][0] - coordinates[3][0])
        right_height = ln.norm(coordinates[1][1] - coordinates[3][1])

        bias_width = upper_width - lower_width
        bias_height = left_height - right_height

        if bias_width <= 0:
            width = lower_width
        else:
            width = upper_width

        if bias_height <= 0:
            height = right_height
        else:
            height = left_height

    return width, height, coordinates
                       #(grid_edge_points, vert_distance, vert_max_points, top_max_points_shape, hor_distance, hor_max_points, left_max_points_shape, iou_or)
# def distance_visualiser(grid_edge_points, ver_pixel_coordinates, hor_pixel_coordinates, img_or, vert_distance_list, hor_distance_list):
# def distance_visualiser(grid_edge_points, vert_distance, vert_max_points, top_max_points_shape, hor_distance, hor_max_points, left_max_points_shape, img_or):
def distance_visualiser(grid_edge_points, top_max_points, bottom_max_points, left_max_points, right_max_points, top_distance_list, bottom_distance_list, left_distance_list, right_distance_list, img_or, img_name, no_substraction=True,path=''):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 0, 255) 
    font_thickness = 2
    
    #VERTICAL 
    for i, (point2, text) in enumerate(zip(top_max_points, top_distance_list)):
        if not no_substraction:    
            if i%2 == 0:
                point1 = (point2[0], grid_edge_points[0])
            else:
                point1 = (point2[0], grid_edge_points[0]+14)
        else: 
            point1 = (point2[0], grid_edge_points[0])
        cv2.line(img_or, point2, point1, [0, 0, 255], 2)
        text  = f'{text:.1f}mm'
        cv2.putText(img_or, text, (point1[0]+2, point1[1] + int((point2[1]-point1[1])/2)), font, font_scale, font_color, font_thickness)
    for i, (point2, text) in enumerate(zip(bottom_max_points, bottom_distance_list)):
        if not no_substraction:    
            if i%2 == 0:
                point1 = (point2[0], grid_edge_points[2])
            else:
                point1 = (point2[0], grid_edge_points[2]-14)
        else:
            point1 = (point2[0], grid_edge_points[2])
        cv2.line(img_or, point2, point1, [0, 0, 255], 2)
        text  = f'{text:.1f}mm'
        cv2.putText(img_or, text, (point1[0]+2, point1[1] + int((point2[1]-point1[1])/2)), font, font_scale, font_color, font_thickness)
    
    #HORIZONTAL
    #for point1, point2, text in zip(hor_pixel_coordinates[0::2], hor_pixel_coordinates[1::2], hor_distance_list): 
    for i, (point2, text) in enumerate(zip(left_max_points, left_distance_list)):
        if not no_substraction:
            if i%2 == 0:
                point1 = (grid_edge_points[1], point2[1])
            else:
                point1 = (grid_edge_points[1]+14, point2[1])
        else:
            point1 = (grid_edge_points[1], point2[1])
        cv2.line(img_or, point2, point1, [0, 0, 255], 2)
        text  = f'{text:.1f}mm'
        cv2.putText(img_or, text, (point2[0]+2, point2[1] + int((point1[1]-point2[1])/2)), font, font_scale, font_color, font_thickness)
    
    for i, (point2, text) in enumerate(zip(right_max_points, right_distance_list)): 
        if not no_substraction:
            if i%2 == 0:
                point1 = (grid_edge_points[3], point2[1])
            else:
                point1 = (grid_edge_points[3]-14, point2[1])
        else:
            point1 = (grid_edge_points[3], point2[1])
        cv2.line(img_or, point2, point1, [0, 0, 255], 2)
        text  = f'{text:.1f}mm'
        cv2.putText(img_or, text, (point2[0]+2, point2[1] + int((point1[1]-point2[1])/2)), font, font_scale, font_color, font_thickness)
    cv2.imwrite(path + f'/returnpoints_images/{img_name}_RP.jpg', iou_or)
    # test = Image.fromarray(img_or)
    # test.show()
#%%
#Input final padded image 
# fp_img_arr = cv2.imread('/Users/menelaos/Desktop/3d_printing_project/PCL_ML-FA11_X100_FMEW_grid_05_final_padding.png', cv2.IMREAD_GRAYSCALE)
#dst_rgb = cv2.imread('/Users/menelaos/Desktop/3d_printing_project/returnbatch2/PCL_ML-FA28_X100_FMEW_grid_04_final_padding.png')
# iou_img_arr = cv2.imread('/Users/menelaos/Desktop/3d_printing_project/IoU/PCL_ML-FA13_X100_FMEW_grid_04_IoU.jpg', cv2.IMREAD_GRAYSCALE)
# sobel_dy = cv2.imread('/Users/menelaos/Desktop/3d_printing_project/sobel.png', cv2.IMREAD_GRAYSCALE)
# sobel_dx = cv2.imread('/Users/menelaos/Desktop/3d_printing_project/sobel_dx.png', cv2.IMREAD_GRAYSCALE)


path = os.getcwd()
final_padded_dir = path + '/returnbatch1'
iou_dir = path + '/IoU'
# rgb_search_path = "/Users/menelaos/LRZ Sync+Share/FMEWgoesML (Annika Hangleiter)/202309_NewSession/15x15_grid_new session_2nd_printer/01_Images_15x15_grids/"
fp_images = os.listdir(final_padded_dir)
iou_images = os.listdir(iou_dir)
return_points = []
for fp_img in fp_images:
    if fp_img == '.DS_Store':
        continue
    else:
        print('fp img', fp_img)
        fp_image_num = fp_img.split('_')[1]
        fp_grid_num = fp_img.split('_')[5]
    for iou_img in iou_images:
        # print('iou_img', iou_img)
        if iou_img == '.DS_Store':
            continue
        else:
            iou_image_num = iou_img.split('_')[1]
            iou_grid_num = iou_img.split('_')[5]
        
        if fp_image_num == iou_image_num and fp_grid_num==iou_grid_num:
            top_distance_list, bottom_distance_list, right_distance_list, left_distance_list = [],[],[],[]
            top_max_points, bottom_max_points, right_max_points, left_max_points = [],[],[],[]
            one_time_process_executed = False
            
            # search_pattern = f"{rgb_search_path}/**/{filename}"
            # dst_rgb = glob.glob(search_pattern, recursive=True)

            fp_img_arr = cv2.imread(f'{final_padded_dir}/{fp_img}', cv2.IMREAD_GRAYSCALE)
            iou_img_arr = cv2.imread(f'{iou_dir}/{iou_img}', cv2.IMREAD_GRAYSCALE)
            iou_or = cv2.imread(f'{iou_dir}/{iou_img}')
            
            grid_arr = cv2.imread(f'/Users/menelaos/LRZ Sync+Share/FMEWgoesML (Annika Hangleiter)/ideal_grids/Grid_{fp_grid_num}.png', cv2.IMREAD_GRAYSCALE)
            grid_arr = cv2.resize(grid_arr, dsize = (2000,2000))
            
            print('fp image num',fp_image_num)
            print('iou image num',iou_image_num)
            print('fp grid num', fp_grid_num)
            print('iou grid num', iou_grid_num)
            fp_img_arr = np.array(fp_img_arr)
            iou_img_arr = np.array(iou_img_arr)
            
            # test = Image.fromarray(fp_img_arr)
            # test.show()
            # test1 = Image.fromarray(iou_img_arr)
            # test1.show()
            
            fp_img_arr = fp_img_arr/255
            fp_img_arr[fp_img_arr < 0.8] = 0
            fp_img_arr[fp_img_arr > 0.8] = 1

            grid_arr = grid_arr/255
            grid_arr[grid_arr < 0.8] = 0
            grid_arr[grid_arr > 0.8] = 1

            iou_img_arr = iou_img_arr/255
            iou_img_arr[iou_img_arr < 0.3] = 0
            iou_img_arr[iou_img_arr > 0.3] = 1
            grid_edge_points = []
            for i in range(int(iou_img_arr.shape[0]/3)):
                for j in range(int(iou_img_arr.shape[1]/3)):
                    if_break = False
                    if iou_img_arr[i][j] == 1:
                        # top_x = j
                        top = i
                        grid_edge_points.append(top)
                        if_break = True
                        print('top:', top)
                        # print(top_y)
                        break
                if if_break == True:
                    break 
                
            for i in range(int(iou_img_arr.shape[1]/3)):
                for j in range(int(iou_img_arr.shape[0]/3)):
                    if_break = False
                    if iou_img_arr[j][i] == 1:
                        left = i
                        grid_edge_points.append(left)
                        # left_y = j
                        if_break = True
                        print('left:', left)
                        # print(left_y)
                        break
                if if_break == True:
                    break 
                
            for i in reversed(range(iou_img_arr.shape[0])):
                for j in reversed(range(iou_img_arr.shape[1])):
                    if_break = False
                    if iou_img_arr[i][j] == 1 and j >= int(iou_img_arr.shape[0]/2):
                        # bottom_x = j
                        bottom = i
                        grid_edge_points.append(bottom)
                        if_break = True
                        print('bottom:', bottom)
                        # print(bottom_y)
                        break
                if if_break == True:
                    break 
                
            for i in reversed(range(iou_img_arr.shape[1])):
                for j in reversed(range(iou_img_arr.shape[0])):
                    if_break = False
                    if iou_img_arr[j][i] == 1 and j <= int(iou_img_arr.shape[1]/2):
                        right = i
                        grid_edge_points.append(right)
                        # right_y = j
                        if_break = True
                        print('right:', right)
                        # print(right_y)
                        break
                if if_break == True:
                    break 
                
            print("grid edge points:", grid_edge_points)  
            a, b, dst_coordinates =  cross_search(fp_img_arr, 20, 'Image')
            print('coordinates:', dst_coordinates)
            # dst_coordinates =[(354, 355), (1618, 345), (364, 1658), (1631, 1648)]
            offset = 40

            #### FILTER OUT THE IRRELEVANT PARTS FOR THE 4 AREAS OF THE IMAGE ####
            dst_top, dst_bottom, dst_right, dst_left = np.copy(fp_img_arr), np.copy(fp_img_arr), np.copy(fp_img_arr), np.copy(fp_img_arr)

            dst_top[(dst_coordinates[0][1]-offset):, :] = 0
            # test1 = Image.fromarray(dst_top*255)
            # test1.show()

            dst_bottom[:(dst_coordinates[3][1]+offset), :] = 0
            # dst_coordinates[3][1]
            # test2 = Image.fromarray(dst_bottom*255)
            # test2.show()

            dst_left[:,(dst_coordinates[0][0]-offset):] = 0
            # test3 = Image.fromarray(dst_left*255)
            # test3.show()

            dst_right[:, :(dst_coordinates[3][0]+offset)] = 0
            # test4 = Image.fromarray(dst_right*255)
            # test4.show()
            #########################################################################################################################################################################################################
            ########################################################################################################################################################################################################
            #### ITERATE THROUGH THE 4 AREAS #####
            one_time_process_executed == False
            areas = ['top', 'bottom', 'left', 'right']
            area_iter = 0
            for area in areas:
                area_iter += 1 
                else_top_iter, else_bottom_iter, else_right_iter, else_left_iter = 0, 0, 0, 0
                if_top_iter, if_bottom_iter, if_right_iter, if_left_iter = 0, 0, 0, 0
                print('area', area)
                array_name = f'dst_{area}' 
                grid_name = f'{area}' 

                temp_area = globals()[array_name] 
                grid_area = globals()[grid_name] 

                # print('grid_area', grid_area)

                image_sep = measure.label(temp_area)

                for i in np.unique(image_sep, return_counts=False):
                        if np.count_nonzero(image_sep == i) < 200:
                            temp_area[image_sep==i] = 0

                image_sep = measure.label(temp_area)
                unique_el = list(np.unique(image_sep))
                unique_el.pop(0)
                #####################################################################################################################################################################
                #####################################################################################################################################################################
                if len(unique_el) < 10:
                    no_substraction = False
                    print('if unique elements:', unique_el)

                    if not one_time_process_executed:
                        print('enter')
                        sobel_dx = scipy.ndimage.sobel(fp_img_arr,0)
                        sobel_dx[sobel_dx > 0] =1 
                        image_sep = measure.label(sobel_dx)

                        for i in np.unique(image_sep, return_counts=False):
                                if np.count_nonzero(image_sep == i) < 30:
                                    sobel_dx[image_sep==i] = 0
                        # cv2.imwrite('/Users/menelaos/Desktop/3d_printing_project/sobel.png', sobel_dx*255)
                        test= Image.fromarray(sobel_dx*255)
                        test.show()

                        sobel_dy = scipy.ndimage.sobel(fp_img_arr)
                        sobel_dy[sobel_dy>0]=1 
                        image_sep = measure.label(sobel_dy)

                        for i in np.unique(image_sep, return_counts=False):
                                if np.count_nonzero(image_sep == i) < 30:
                                    sobel_dy[image_sep==i] = 0
                        # cv2.imwrite('/Users/menelaos/Desktop/3d_printing_project/sobel_dx.png', sobel_dy*255)
                        test= Image.fromarray(sobel_dy*255)
                        test.show()
                        use_areas = False
                    
                        barriers_offset = 10
                        xbarriers_redo = True
                        xredo = 0
                        failed = False
                        while xbarriers_redo:
                            xbarriers = []
                            if xredo < int(sobel_dx.shape[0]/2) - dst_coordinates[0][1]:
                                xredo += 1
                            else:
                                failed = False
                                break
                            for j in range(dst_coordinates[0][1] - barriers_offset, dst_coordinates[2][1] + barriers_offset):
                                diff = abs(sobel_dx[j][int(sobel_dx.shape[0]/2)+xredo] - sobel_dx[j+1][int(sobel_dy.shape[0]/2)+xredo])

                                if diff == 1:                                       
                                    xbarriers.append(j)
                            xbarriers = np.array(xbarriers)
                            xbarriers_mean = []
                            xbarrier_count = 0
                            for i in range(xbarriers.shape[0]-1):
                                    if i == 0:
                                        diff = abs(xbarriers[i] - xbarriers[i+1])
                                        if diff < 5:
                                            xmean = int((xbarriers[i] + xbarriers[i+1])/2)
                                            xbarriers_mean.append(xmean)
                                        else:
                                            xmean = xbarriers[i]
                                            xbarriers_mean.append(xmean)   

                                    else:
                                        diff1 = abs(xbarriers[i] - xbarriers[i+1])
                                        diff2 = abs(xbarriers[i-1] - xbarriers[i])
                                        # if diff1< 5 and diff2< 5:

                                        #     xmean = int((xbarriers[i] + xbarriers[i+1])/2)
                                        #     xbarriers_mean.append(xmean)

                                        if diff1< 5 and diff2> 5 and xbarrier_count == 0:
                                            xbarrier_count += 1
                                            xmean = int((xbarriers[i] + xbarriers[i+1])/2)
                                            xbarriers_mean.append(xmean)

                                        elif diff1> 5 and diff2< 5:
                                            xbarrier_count =0
                                            continue

                                        elif diff1> 5 and diff2> 5:
                                            xbarrier_count =0
                                            xbarriers_mean.append(xbarriers[i])
                                        else: continue
                            print('xbarriers',xbarriers)
                            print('xbarriers_mean', xbarriers_mean)
                            if len(xbarriers_mean) == 34:
                                print('success xbarries', int(sobel_dx.shape[0]/2)+xredo)
                                xbarriers_redo = False
                            
                        xbarriers_mean = np.array(xbarriers_mean)
                        print('xbarriers',xbarriers)
                        print('xbarriers_mean', xbarriers_mean)
                        xbarriers_mean_left = xbarriers_mean
                        xbarriers_mean_right = xbarriers_mean

                        barriers_offset = 10
                        ybarriers_redo = True
                        yredo = 0
                        while ybarriers_redo:
                            if failed: break 
                            ybarriers = []
                            yredo += 1
                            for j in range(dst_coordinates[0][0] - barriers_offset, dst_coordinates[1][0] + barriers_offset):
                                diff = abs(sobel_dy[int(sobel_dy.shape[0]/2) +yredo][j] - sobel_dy[int(sobel_dy.shape[0]/2) + yredo][j+1])

                                if diff == 1:
                                    ybarriers.append(j)

                            ybarriers = np.array(ybarriers)
                            ybarriers_mean = []
                            ybarrier_count = 0
                            for i in tqdm(range(ybarriers.shape[0]-1)):
                                if i == 0:
                                    diff = abs(ybarriers[i] - ybarriers[i+1])
                                    if diff < 5:
                                        ymean = int((ybarriers[i] + ybarriers[i+1])/2)
                                        ybarriers_mean.append(ymean)
                                    else:
                                        ymean = ybarriers[i]
                                        ybarriers_mean.append(ymean)   

                                else:
                                    diff1 = abs(ybarriers[i] - ybarriers[i+1])
                                    diff2 = abs(ybarriers[i-1] - ybarriers[i])
                                    # if diff1< 5 and diff2< 5:
                                    #     ymean = int((ybarriers[i] + ybarriers[i+1])/2)
                                    #     ybarriers_mean.append(ymean)

                                    if diff1< 5 and diff2> 5 and ybarrier_count == 0:
                                        ybarrier_count += 1
                                        ymean = int((ybarriers[i] + ybarriers[i+1])/2)
                                        ybarriers_mean.append(ymean)

                                    elif diff1> 5 and diff2< 5:
                                        ybarrier_count =0
                                        continue

                                    elif diff1> 5 and diff2> 5:
                                        ybarrier_count =0
                                        ybarriers_mean.append(ybarriers[i])
                                    else: continue
                            if len(ybarriers_mean) == 34:
                                print('success ybarries', int(sobel_dy.shape[0]/2)+yredo)
                                ybarriers_redo = False
                        ybarriers_mean = np.array(ybarriers_mean)
                        print('ybarriers',ybarriers)
                        print('ybarriers_mean', ybarriers_mean)
                        ybarriers_mean_top = ybarriers_mean
                        ybarriers_mean_bottom = ybarriers_mean
                    
                        if failed:
                            barriers_offset = 10
                            left_redo = True
                            l = 0
                            while left_redo:
                                xbarriers_left = []
                                l += 1
                                for j in tqdm(range(dst_coordinates[0][1] - barriers_offset, dst_coordinates[2][1] + barriers_offset)):
                                    diff_left = abs(sobel_dx[j][dst_coordinates[0][0]+l] - sobel_dx[j+1][dst_coordinates[0][0]+l])

                                    if diff_left == 1:
                                        xbarriers_left.append(j)

                                xbarriers_left = np.array(xbarriers_left)

                                xbarriers_mean_left = []
                                barrier_left_count = 0
                                for i in tqdm(range(xbarriers_left.shape[0]-1)):
                                    if i == 0:
                                    
                                        diff = abs(xbarriers_left[i] - xbarriers_left[i+1])
                                        if diff < 5:
                                        
                                            xmean_left = int((xbarriers_left[i] + xbarriers_left[i+1])/2)
                                            xbarriers_mean_left.append(xmean_left)
                                        else:
                                            xmean_left = xbarriers_left[i]
                                            xbarriers_mean_left.append(xmean_left)   
                                    else:
                                        diff1 = abs(xbarriers_left[i] - xbarriers_left[i+1])
                                        diff2 = abs(xbarriers_left[i-1] - xbarriers_left[i])
                                        # if diff1< 5 and diff2< 5 : 

                                        #     xmean_left = int((xbarriers_left[i] + xbarriers_left[i+1])/2)
                                        #     xbarriers_mean_left.append(xmean_left)

                                        if diff1< 5 and diff2> 10 and barrier_left_count == 0:
                                            barrier_left_count += 1
                                            xmean_left = int((xbarriers_left[i] + xbarriers_left[i+1])/2)
                                            xbarriers_mean_left.append(xmean_left)

                                        elif diff1> 5 and diff2< 5:
                                            barrier_left_count =0
                                            continue
                                        
                                        elif diff1> 5 and diff2> 5:
                                            barrier_left_count =0
                                            xbarriers_mean_left.append(xbarriers_left[i])

                                        else: continue
                                if len(xbarriers_mean_left) == 34 :
                                    print('success left', dst_coordinates[0][0]+l)
                                    left_redo = False

                            right_redo = True
                            r = 0
                            while right_redo:
                                xbarriers_right = []
                                r += 1
                                for j in range(dst_coordinates[0][1] - barriers_offset, dst_coordinates[2][1] + barriers_offset):
                                    diff_right = abs(sobel_dx[j][dst_coordinates[1][0]-r] - sobel_dx[j+1][dst_coordinates[1][0]-r])

                                    if diff_right == 1:
                                        xbarriers_right.append(j)

                                xbarriers_right = np.array(xbarriers_right)
                                xbarriers_mean_right = []
                                barrier_right_count = 0
                                for i in range(xbarriers_right.shape[0]-1):
                                    if i == 0:
                                        diff = abs(xbarriers_right[i] - xbarriers_right[i+1])
                                        if diff < 5:
                                            xmean_right = int((xbarriers_right[i] + xbarriers_right[i+1])/2)
                                            xbarriers_mean_right.append(xmean_right)
                                        else:
                                            xmean_right = xbarriers_right[i]
                                            xbarriers_mean_right.append(xmean_right)   
                                    else:
                                        diff1 = abs(xbarriers_right[i] - xbarriers_right[i+1])
                                        diff2 = abs(xbarriers_right[i-1] - xbarriers_right[i])
                                        # if diff1< 5 and diff2< 5:

                                        #     xmean_right = int((xbarriers_right[i] + xbarriers_right[i+1])/2)
                                        #     xbarriers_mean_right.append(xmean_right)

                                        if diff1< 5 and diff2> 5 and barrier_right_count == 0:
                                            barrier_right_count += 1
                                            xmean_right = int((xbarriers_right[i] + xbarriers_right[i+1])/2)
                                            xbarriers_mean_right.append(xmean_right)

                                        elif diff1> 5 and diff2< 5:
                                            barrier_right_count =0
                                            continue
                                        
                                        elif diff1> 5 and diff2> 5:
                                            barrier_right_count =0
                                            xbarriers_mean_right.append(xbarriers_right[i])
                                        else: continue
                                if len(xbarriers_mean_right) == 34:
                                    print('success right', dst_coordinates[1][0]-r)
                                    right_redo = False


                                # if diff1 >5 and counter < 1:
                                #     counter +=1
                                #     # print('i', i)
                                #     # print('xbarriers',xbarriers[i])
                                #     xbarriers_mean.append(xbarriers[i])
                                # elif diff <5 and diff>-5:
                                #     counter = 0
                                #     xmean = int((xbarriers[i] + xbarriers[i+1])/2)
                                #     # print('i', i)
                                #     # print('xmean', xmean)
                                #     xbarriers_mean.append(xmean)
                                # else:
                                #     continue
                            xbarriers_mean_left = np.array(xbarriers_mean_left)
                            xbarriers_mean_right = np.array(xbarriers_mean_right)
                            print('xbarriers left:',xbarriers_left)
                            print('xbarriers right:',xbarriers_right)
                            print('xbarriers_mean left:', xbarriers_mean_left)
                            print('xbarriers_mean right:', xbarriers_mean_right)

                            top_redo = True
                            t = 0
                            while top_redo:
                                ybarriers_top = []
                                t += 1
                                for j in range(dst_coordinates[0][0] - barriers_offset, dst_coordinates[1][0] + barriers_offset):
                                    diff_right = abs(sobel_dy[dst_coordinates[1][0]+t][j] - sobel_dy[dst_coordinates[1][0]+t][j+1])

                                    if diff_right == 1:
                                        ybarriers_top.append(j)

                                ybarriers_top = np.array(ybarriers_top)
                                ybarriers_mean_top = []
                                barrier_top_count = 0
                                for i in range(ybarriers_top.shape[0]-1):
                                    if i == 0:
                                        diff = abs(ybarriers_top[i] - ybarriers_top[i+1])
                                        if diff < 5:
                                            ymean_top = int((ybarriers_top[i] + ybarriers_top[i+1])/2)
                                            ybarriers_mean_top.append(ymean_top)
                                        else:
                                            ymean_top = ybarriers_top[i]
                                            ybarriers_mean_top.append(ymean_top)   
                                    else:
                                        diff1 = abs(ybarriers_top[i] - ybarriers_top[i+1])
                                        diff2 = abs(ybarriers_top[i-1] - ybarriers_top[i])
                                        # if diff1< 5 and diff2< 5:

                                        #     ymean_top = int((ybarriers_top[i] + ybarriers_top[i+1])/2)
                                        #     ybarriers_mean_top.append(ymean_top)

                                        if diff1< 5 and diff2> 5 and barrier_top_count == 0:
                                            barrier_top_count += 1
                                            ymean_top = int((ybarriers_top[i] + ybarriers_top[i+1])/2)
                                            ybarriers_mean_top.append(ymean_top)

                                        elif diff1> 5 and diff2< 5:
                                            barrier_top_count =0
                                            continue
                                        
                                        elif diff1> 5 and diff2> 5:
                                            barrier_top_count =0
                                            ybarriers_mean_top.append(ybarriers_top[i])
                                        else: continue
                                if len(ybarriers_mean_top) == 34:
                                    print('success top', dst_coordinates[1][0]+t)
                                    top_redo = False
                            ybarriers_mean_top = np.array(ybarriers_mean_top)

                            bottom_redo = True
                            t = 0
                            while bottom_redo:
                                ybarriers_bottom = []
                                t += 1
                                for j in range(dst_coordinates[0][0] - barriers_offset, dst_coordinates[1][0] + barriers_offset):
                                    diff_right = abs(sobel_dy[dst_coordinates[2][0]-t][j] - sobel_dy[dst_coordinates[3][0]-t][j+1])

                                    if diff_right == 1:
                                        ybarriers_bottom.append(j)

                                ybarriers_bottom = np.array(ybarriers_bottom)
                                ybarriers_mean_bottom = []
                                barrier_bottom_count = 0
                                for i in range(ybarriers_bottom.shape[0]-1):
                                    if i == 0:
                                        diff = abs(ybarriers_bottom[i] - ybarriers_bottom[i+1])
                                        if diff < 5:
                                            ymean_bottom = int((ybarriers_bottom[i] + ybarriers_bottom[i+1])/2)
                                            ybarriers_mean_bottom.append(ymean_bottom)
                                        else:
                                            ymean_bottom = ybarriers_bottom[i]
                                            ybarriers_mean_bottom.append(ymean_bottom)   
                                    else:
                                        diff1 = abs(ybarriers_bottom[i] - ybarriers_bottom[i+1])
                                        diff2 = abs(ybarriers_bottom[i-1] - ybarriers_bottom[i])
                                        # if diff1< 5 and diff2< 5:

                                        #     ymean_bottom = int((ybarriers_bottom[i] + ybarriers_bottom[i+1])/2)
                                        #     ybarriers_mean_bottom.append(ymean_bottom)

                                        if diff1< 5 and diff2> 5 and barrier_bottom_count == 0:
                                            barrier_bottom_count += 1
                                            ymean_bottom = int((ybarriers_bottom[i] + ybarriers_bottom[i+1])/2)
                                            ybarriers_mean_bottom.append(ymean_bottom)

                                        elif diff1> 5 and diff2< 5:
                                            barrier_bottom_count =0
                                            continue
                                        
                                        elif diff1> 5 and diff2> 5:
                                            barrier_bottom_count =0
                                            ybarriers_mean_bottom.append(ybarriers_bottom[i])
                                        else: continue
                                if len(ybarriers_mean_bottom) == 34:
                                    print('success bottom', dst_coordinates[1][0]+t)
                                    bottom_redo = False
                            ybarriers_mean_bottom = np.array(ybarriers_mean_bottom)

                            print('ybarriers top:',ybarriers_top)
                            print('ybarriers bottom:',ybarriers_bottom)
                            print('ybarriers_mean_top:', ybarriers_mean_top)  
                            print('ybarriers_mean_bottom', ybarriers_mean_bottom)  
                        one_time_process_executed = True    

#############################################################################################################################################################################################
                        # barriers_offset = 10
                        # ybarriers = []
                        # for j in range(dst_coordinates[0][0] - barriers_offset, dst_coordinates[1][0] + barriers_offset):
                        #     diff = abs(sobel_dy[int(sobel_dy.shape[0]/2)][j] - sobel_dy[int(sobel_dy.shape[0]/2)][j+1])
                        #     print('diff', diff)
                        #     if diff == 1:
                        #         ybarriers.append(j)
                        # ybarriers1 = ybarriers[0::2]
                        # ybarriers2 = ybarriers[1::2]

                        # ybarriers1 = np.array(ybarriers1)
                        # ybarriers2 = np.array(ybarriers2)
                        # ybarriers = (ybarriers2 + ybarriers1)/2
                        # ybarriers = ybarriers.astype(int)
                        # print('ybarries', ybarriers)
                        # test = Image.fromarray(temp_area*255)
                        # test.show()


                    ##########################################################################################################################################
                    ##########################################################################################################################################
                    if area == 'top':
                        print('iftop')
                        if_top_iter += 1
                        #### SOBEL IDEA
                        # test = Image.fromarray(temp_area*255)
                        # test.show()
                        # sobel_dx = scipy.ndimage.sobel(temp_area, 0)
                        # test= Image.fromarray(sobel_dx*255)
                        # test.show()
                        # image_sep = measure.label(sobel_dx)

                        # for i in np.unique(image_sep, return_counts=False):
                        #     if np.count_nonzero(image_sep == i) < 10:

                        #         sobel_dx[image_sep==i] = 0
                                    # test = Image.fromarray(sobel_dx*255)
                        # test.show()
                        # if_break = False
                        iter_offset = 4
                        top_max_points = []

                        for k in tqdm(range(ybarriers_mean_top.shape[0]-1)):
                            if_break = False
                            for i in tqdm(range(temp_area.shape[0])):    
                                for j in tqdm(range(ybarriers_mean_top[k]+6, ybarriers_mean_top[k+1]-6)):#+iter_offset
                                    iou_or[:, ybarriers_mean_top[k]+1] = [255, 255 ,0]
                                    if temp_area[i][j] == 1:
                                        # print('i', i)
                                        # print('j', j)
                                        top_distance =  i - top
                                        top_distance_list.append(top_distance)
                                        top_max_points.append([j, i])
                                        if_break = True
                                        break
                                if if_break == True:
                                    break
                        print('top_max_points', top_max_points)
                        # test = Image.fromarray(dst_rgb)
                        # test.show()     
                    elif area == 'left':
                        if_left_iter += 1
                        iter_offset = 5
                        left_max_points = []
                        for k in range(xbarriers_mean_left.shape[0]-1):
                            if_break = False
                            for i in range(temp_area.shape[1]):    
                                for j in range(xbarriers_mean_left[k] + iter_offset, xbarriers_mean_left[k+1] - iter_offset):
                                    iou_or[xbarriers_mean_left[k], :] = [0, 255 , 0]
                                    # dst_rgb[j][i] = [255, 192 ,203]
                                    if temp_area[j][i] == 1:
                                        # print('i', i)
                                        # print('j', j)
                                        left_distace = i - left
                                        left_distance_list.append(left_distace)
                                        left_max_points.append([i, j])
                                        if_break = True
                                        break
                                if if_break == True:
                                    break
                        print('left_max_points', left_max_points)
                        # test = Image.fromarray(dst_rgb)
                        # test.show()    

                    elif area == 'bottom':

                        iter_offset = 6
                        bottom_max_points = []
                        for k in range(ybarriers_mean_bottom.shape[0]-1):
                            if_break = False
                            for i in reversed(range(temp_area.shape[0])):    
                                for j in range(ybarriers_mean_bottom[k]+iter_offset, ybarriers_mean_bottom[k+1]-iter_offset):
                                    iou_or[:, ybarriers_mean_bottom[k]] = [160, 32, 240]
                                    if temp_area[i][j] == 1:
                                        # print('i', i)
                                        # print('j', j)
                                        bottom_distance = bottom - i 
                                        bottom_distance_list.append(bottom_distance)
                                        bottom_max_points.append([j, i])
                                        if_break = True
                                        break
                                if if_break == True:
                                    break
                        print('bottom_max_points', bottom_max_points)

                    elif area == 'right':

                        iter_offset = 6
                        right_max_points = []
                        for k in range(xbarriers_mean_right.shape[0]-1):
                            if_break = False
                            for i in reversed(range(temp_area.shape[1])):    
                                for j in range(xbarriers_mean_right[k] + iter_offset, xbarriers_mean_right[k+1] - iter_offset):
                                    iou_or[xbarriers_mean_right[k], :] = [255, 0 ,0]
                                    if temp_area[j][i] == 1:
                                        # print('i', i)
                                        # print('j', j)
                                        right_distance = right - i 
                                        right_distance_list.append(right_distance)
                                        right_max_points.append([i, j])
                                        if_break = True
                                        break
                                if if_break == True:
                                    break
                        print('right_max_points', right_max_points)

                ####### UniqueElements> 10
                #####################################################################################################################################################################
                #####################################################################################################################################################################
                else:
                    no_substraction = True
                    print('else unique elements:', unique_el)
                     
                    for i in unique_el:
                        if_break = False 
                        area_copy = np.copy(temp_area)
                        area_copy[image_sep!=i] = 0
                        # test= Image.fromarray(area_copy*255)
                        # test.show()  
                        if area == 'top':
                            else_top_iter += 1
                            for i in range(area_copy.shape[0]):
                                for j in range(area_copy.shape[1]):
                                    if_break = False
                                    if area_copy[i][j] == 1:
                                        top_distance = i - top   
                                        top_distance_list.append(top_distance)
                                        top_max_points.append([j, i])
                                        # print(i)
                                        # print(j)
                                        if_break = True
                                        break
                                if if_break == True:
                                    break   
                            

                        elif area == 'left':
                            else_left_iter += 1
                            for i in range(area_copy.shape[1]):
                                for j in range(area_copy.shape[0]):
                                    if_break = False 
                                    if area_copy[j][i] == 1:
                                        left_distance = i - left    
                                        left_distance_list.append(left_distance)
                                        left_max_points.append([i, j])
                                        # print(i)
                                        # print(j)
                                        if_break = True
                                        break
                                if if_break == True:
                                    break
                                
                        
                        elif area == 'bottom':
                            else_bottom_iter += 1
                            for i in reversed(range(area_copy.shape[0])):
                                for j in reversed(range(area_copy.shape[1])):
                                    if_break = False
                                    if area_copy[i][j] == 1:
                                        bottom_distance = bottom - i 
                                        bottom_distance_list.append(bottom_distance)
                                        bottom_max_points.append([j, i])
                                        # print(i)
                                        # print(j)
                                        if_break = True
                                        break
                                if if_break == True:
                                    break
                            
                        elif area == 'right':
                            else_right_iter += 1
                            for i in reversed(range(area_copy.shape[1])):
                                for j in reversed(range(area_copy.shape[0])):
                                    if_break = False 
                                    if area_copy[j][i] == 1:
                                        right_distance = right - i 
                                        right_distance_list.append(right_distance)
                                        right_max_points.append([i, j])
                                        # print(i)
                                        # print(j)
                                        if_break = True
                                        break
                                if if_break == True:
                                    break   
                            
                # else_iter = else_top_iter+else_bottom_iter+else_left_iter+else_right_iter
                # if_iter = if_top_iter+if_bottom_iter+if_left_iter+if_right_iter
                # if area_iter == 4 and else_iter >= if_iter and else_iter<4 and if_iter < 4:
                #     else_list = [ else_top_iter, else_bottom_iter, else_left_iter, else_right_iter]
                #     for i in range(len(else_list)):
                #         if i != 0:
                #             else_list.pop(i)
                #     print('else list ',else_list)  
            top_distance_list, left_distance_list, bottom_distance_list, right_distance_list = np.array(top_distance_list), np.array(left_distance_list), np.array(bottom_distance_list), np.array(right_distance_list)
            print('top_distance _list', top_distance_list)
            print('top_max_points', top_max_points)
            
            print('left_distance_list', left_distance_list)
            print('left_max_points', left_max_points)
            
            print('bottom_distance_list', bottom_distance_list)
            print('bottom_max_points', bottom_max_points)
            
            print('right_distance_list', right_distance_list)
            print('right_max_points', right_max_points)
            top_max_points, left_max_points, bottom_max_points, right_max_points = np.array(top_max_points), np.array(left_max_points), np.array(bottom_max_points), np.array(right_max_points)
            if not no_substraction:
                top_distance_list[1::2] -= 14
                left_distance_list[1::2] -= 14
                bottom_distance_list[1::2] -= 14
                right_distance_list[1::2] -= 14
 
            top_distance_list = (top_distance_list / 110) 
            left_distance_list = (left_distance_list / 110) 
            bottom_distance_list = (bottom_distance_list / 110) 
            right_distance_list = (right_distance_list / 110) 

            vert_distance = np.concatenate([top_distance_list, bottom_distance_list])
            hor_distance = np.concatenate([left_distance_list, right_distance_list])

            vert_max_points = np.concatenate([top_max_points, bottom_max_points])
            hor_max_points = np.concatenate([left_max_points, right_max_points])

            vert_mean = np.mean(vert_distance)
            vert_std = np.std(vert_distance)
            hor_mean = np.mean(hor_distance)
            hor_std = np.std(hor_distance)
            name = f"{fp_image_num}_{fp_grid_num}"

            img_return_point = f'{name}, {vert_mean}, {vert_std}, {hor_mean}, {hor_std}'
            return_points.append(img_return_point)

            distance_visualiser(grid_edge_points, top_max_points, bottom_max_points, left_max_points, right_max_points, top_distance_list, bottom_distance_list, left_distance_list, right_distance_list, iou_or, name, no_substraction=no_substraction, path=path)

            file= open('returnpoints_new.txt','w')
            for item in return_points:
                file.write(item + "\n")
            file.close()
            
            csv_filename = 'output1.csv'

            with open(csv_filename, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)

                # Write the header row
                csv_writer.writerow(['Image_Point', 'Vertical_Mean', 'Vertical_Std', 'Horizontal_Mean', 'Horizontal_Std'])

                # Iterate through each data point
                for data_point in return_points:
                    # Split the data point based on ':'
                    split_data = data_point.split(',')

                    # Write the data to the CSV file
                    csv_writer.writerow([split_data[0], split_data[1], split_data[2], split_data[3], split_data[4]])

            break   
# a = np.concatenate([top_max_points, bottom_max_points, left_max_points, right_max_points])
# for i in range(a.shape[0]):
#     print(a[i])
#     x = a[i][1]
#     y = a[i][0]
#     dst_rgb[y][x] = [255, 0, 0]
# test = Image.fromarray(dst_rgb)
# test.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Create a sinusoidal wave
x = np.linspace(0, 4 * np.pi, 1000)
y = np.sin(x)

# Find the valleys (minimum points)
valley_indices = np.where((y[:-2] > y[1:-1]) & (y[1:-1] < y[2:]))[0] + 1
valley_points = list(zip(x[valley_indices], y[valley_indices]))

# Plot the sinusoidal wave and highlight the valleys
plt.plot(x, y, label='Sinusoidal Wave')
plt.scatter(*zip(*valley_points), color='red', label='Valleys')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sinusoidal Wave with Valleys')
plt.show()
# %%
import numpy as np

# Example 2D matrix representing a binary image
image_matrix = np.array([[0, 1, 0, 1],
                        [1, 0, 1, 0],
                        [0, 1, 1, 0],
                        [1, 0, 0, 1]])

# Find the coordinates of pixels with value 1
coordinates_ones = np.argwhere(image_matrix == 1)

print("Coordinates of pixels with value 1:")
print(coordinates_ones)

# %%
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

# Create a sine wave signal with variable intervals
x = np.linspace(0, 4 * np.pi, 1000)
y = np.sin(x)

# Find local minima with variable intervals
minima_indices = argrelextrema(y, np.less, order=50)

# Plot the signal and mark the local minima
plt.plot(x, y, label='Sine Wave')
plt.scatter(x[minima_indices], y[minima_indices], color='red', label='Local Minima')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sine Wave with Local Minima and Variable Intervals')
plt.show()
# %%
###### MAYBE USE FOR THE np.unique(areas) < 10 PART
            # test = Image.fromarray(temp_area*255)
            # test.show()
            # y_list = []
            # coordinates_ones = np.argwhere(temp_area == 1)
            # for i in range(coordinates_ones.shape[0]):
            #     print(coordinates_ones[i][:])
            # sorted_points = np.array(sorted(coordinates_ones, key=lambda point: point[1]))
            # for i in range(sorted_points.shape[0]):
            #     print(sorted_points[i][:])
            # for i in range(a.shape[0]):
            #     print(coordinates_ones[i])
            # x = coordinates_ones[:,1]
            # y = coordinates_ones[:,0]
            # counter = 0
            # for i in range(min(x), max(x)+1):
            #     try:
            #         temp = np.argwhere(sorted_points[:, 1] == i)
            #         yvalues = sorted_points[temp]
            #         yvalues = yvalues.reshape(yvalues.shape[0], -1)
            #         yvalues = yvalues[:, 0]
            #         y_mean = int(sum(yvalues)/len(yvalues))
            #         y_list.append(y_mean)
            #         # print(i)
            #         # print(yvalues)
            #         # print('split')
            #     except:   
            #         y_list.append(y_mean)
            #         print(i)
            #         counter +=1
            #         continue
            # print(counter)
            # plotrange = np.arange(min(x),max(x)+1, 1)
            # print(np.array(y_list).shape)
            # print(len(plotrange))
            # y_list = np.array(y_list)
            # minima_indices1 = argrelextrema(y_list, np.less, order = 3)
            # minima_indices2 = argrelextrema(y, np.less, order = 1000)
            
            # plt.figure(1, figsize=(20,8))
            # plt.subplot(1, 2, 1)  
            # plt.scatter(plotrange, y_list, s= 0.2)#sorted_points[:,0]
            # plt.scatter(plotrange[minima_indices1], y_list[minima_indices1], color='red', label='Local Minima')
            
            # plt.figure(2, figsize=(20,8))
            # plt.subplot(1, 2, 2)
            # plt.scatter(x, y, s=0.2)
            # plt.scatter(x[minima_indices1], y[minima_indices1], color='red', label='Local Minima')
            # plt.show()

# %%
############ HOUGH IDEA 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('/Users/menelaos/Desktop/test.png', cv2.IMREAD_GRAYSCALE)


# Apply Gaussian blur to reduce noise and improve circle detection
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Apply Circular Hough Transform
circles = cv2.HoughCircles(
    blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=35, param1=5, param2=15, minRadius=10, maxRadius=33
)

if circles is not None:
    circles = np.uint16(np.around(circles))
    
    # Create an empty mask
    mask = np.zeros_like(image)

    # Draw filled circles on the mask
    for i in circles[0, :]:
        cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), thickness=-1)

    # Apply the mask to the original image
    result_image = cv2.bitwise_and(image, mask)

    plt.figure(figsize=(20,10))
    # Display the original image, the mask, and the result
    plt.subplot(131), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(132), plt.imshow(mask, cmap='gray')
    plt.title('Mask'), plt.xticks([]), plt.yticks([])

    plt.subplot(133), plt.imshow(result_image, cmap='gray')
    plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])

    plt.show()
else:
    print("No circles detected.")
# %%
import numpy as np
import threading
import time
from PIL import Image
def process_chunk(image_sep, temp_area, i, lock):
    with lock:
        if np.count_nonzero(image_sep == i) < 30:
            temp_area[image_sep == i] = 0

def process_image(image_sep, temp_area, num_threads=7):
    unique_values = np.unique(image_sep, return_counts=False)
    lock = threading.Lock()

    def thread_task(chunk_start, chunk_end):
        for i in range(chunk_start, chunk_end):
            process_chunk(image_sep, temp_area, unique_values[i], lock)

    threads = []
    chunk_size = len(unique_values) // num_threads

    for i in range(num_threads):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_threads - 1 else len(unique_values)
        thread = threading.Thread(target=thread_task, args=(start, end))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

fp_img_arr = cv2.imread('/Users/menelaos/Desktop/3d_printing_project/PCL_ML-FA11_X100_FMEW_grid_05_final_padding.png', cv2.IMREAD_GRAYSCALE)
sobel_dx = scipy.ndimage.sobel(fp_img_arr,0)

image_sep = measure.label(sobel_dx)
# Assuming you have already defined image_sep and temp_area
start = time.time()
b = process_image(image_sep, sobel_dx)
end = time.time()
print(end - start )
# test = Image.fromarray(b)
# test.show()
#$$

# %%
path = '/Users/menelaos/Desktop/3dprintannika/redo'
image = cv2.imread(path + '/PCL_MR36_x100_FMEW_grid_01.jpg', cv2.IMREAD_GRAYSCALE)
dim = (2000, 2000)
img = cv2.resize(image, dim)
img = Image.fromarray(img)
img.show()
# %%
a =1.2344
print(f'{a:.1f}')
# %%
