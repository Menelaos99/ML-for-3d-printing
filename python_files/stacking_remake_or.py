#%%
from PIL import Image, ImageOps 
import numpy as np
from skimage.filters.rank import median
from skimage.morphology import disk
from numpy import linalg as ln
from skimage import measure
import cv2
import time
import multiprocessing
from multiprocessing import Lock
from multiprocessing import Process, Value, Array
import os

#%%
def simplify(image, threshold):
    image_copy = np.copy(image)
    image = np.asarray(image_copy)
    im_shape1 = np.shape(image)
    if len(im_shape1) > 2:
        if im_shape1[2] < 4:

            for i in range(im_shape1[0]):
                for j in range(im_shape1[1]):

                    if ln.norm(sum(image[i][j])) < threshold:
                        image[i][j] = [0, 0, 0]
                    else:
                        image[i][j] = [255, 255, 255]           
            new_im = Image.fromarray(image)

        else:
            threshold = threshold + 255
            for i in range(im_shape1[0]):
                for j in range(im_shape1[1]):

                    if ln.norm(sum(image[i][j])) < threshold:
                        image[i][j] = [0, 0, 0, 0]
                    else:
                        image[i][j] = [255, 255, 255, 255] 
            new_im = Image.fromarray(image)

        new_im = new_im.convert('L')
        new_im = np.array(new_im)
        new_im = median(new_im, disk(2))
    
    else: 
        for i in range(im_shape1[0]):
                for j in range(im_shape1[1]):

                    if image[i][j] < threshold:
                        image[i][j] = 0
                    else:
                        image[i][j] = 255
        new_im = Image.fromarray(image)
    
    return new_im

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
                                y_temp = i   
                                x_temp = j  # to move it to the middle 
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
    

    
    print('coordinates', coordinates)

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


def image_resize(image, grid, lw_diff_list):

    padded_grid_parameter = None
    
    if type(image) != "<class 'numpy.ndarray'>":
        image = np.array(image)
    
    grid_shape = np.shape(grid)

    # image = cv2.resize(image, dim)
    # image = (image/255) 
    # image[image<0.2] = 0
    # image[image>0.2] = 1
        
    og_shape = np.shape(image)
    
    #test = Image.fromarray(image*255)
    # test.show()
    
    image_width , image_height, _ = cross_search(image, 20, 'Image')
    
    print('image width', image_width)
    print('image height', image_height)
    
    grid_width , grid_height, _ = cross_search(grid, 20, 'Grid') 
    # print('grid width', grid_width)
    # print('grid height', grid_height)

    # og_im_width = image_edges[1][0] - image_edges[0][0]
    # og_im_length = image_edges[2][1] - image_edges[1][1]

    # grid_im_width = grid_edges[1][0] - grid_edges[0][0]
    # grid_im_length = grid_edges[2][1] - grid_edges[1][1]
    
    ratio_width = grid_width/ image_width 
    ratio_height = grid_height/ image_height

    print('ratio width ' ,ratio_width)
    print('ratio height ' ,ratio_height)

    new_width = int(og_shape[1] * ratio_width)
    new_length = int(og_shape[0] * ratio_height )

    lw_diff = abs(new_width - new_length)
    lw_diff_list.append(lw_diff)

    # print('new_width entire image', new_width)
    # print('new_length entire image', new_length)
    # print('old image', np.shape(image))
    
    #resize images
    new_dim = (new_width, new_length)
    image = image.astype('float32')
    resized_og = cv2.resize(image, new_dim)

    #test_resize.save('/Users/menelaos/Desktop/ba prog/toresize.png')

    resized_og[resized_og<0.3]=0
    resized_og[resized_og>0.3]=1
    
    # resized_og1 = Image.fromarray(resized_og*255)
    # resized_og1.show()
    
    resized_og_shape = np.shape(resized_og)
    
    # resized_image_width, resized_image_height, _ = cross_search(resized_og, 30, 'Image')
    
    # print('new width' ,resized_image_width)
    # print('new height' ,resized_image_height)

    padding_width = grid_shape[1] - resized_og_shape[1]
    padding_height = grid_shape[0] - resized_og_shape[0]

    print('padding_width' ,padding_width)
    print('padding_height' ,padding_height)
    
    if padding_height % 2 == 1 and padding_width % 2 == 1 and padding_width > 0 and padding_height < 0:
        padded_im = np.pad(resized_og, [(0, 0), (int((padding_width/2))+1, int(padding_width/2))], 'constant',  constant_values=(0, 0))
        padded_grid = np.pad(grid, [(int(abs(padding_height)/2 + 1), int(abs(padding_height)/2)), (0, 0)], 'constant',  constant_values=(0, 0))
        padded_grid_parameter = 'Im&Grid'
    
    elif padding_height % 2 == 1 and padding_width % 2 == 1 and padding_width < 0 and padding_height < 0:
        padded_grid = np.pad(grid, [(int(abs(padding_height)/2 + 1), int(abs(padding_height)/2)), (int(abs(padding_width/2))+1, int(abs(padding_width)/2))], 'constant',  constant_values=(0, 0))
        padded_im = resized_og
        padded_grid_parameter = 'OnlyGrid'
    

    elif padding_height % 2 == 1 and padding_width % 2 == 1:
        padded_im = np.pad(resized_og, [(padding_height//2 + 1, padding_height//2), (padding_width//2 + 1 , padding_width//2)], 'constant',  constant_values=(0, 0))

    elif padding_height % 2 == 1 and  padding_width % 2 == 0 and padding_width > 0 and padding_height < 0:
        padded_im = np.pad(resized_og, [(0, 0), (padding_width//2 , padding_width//2)], 'constant',  constant_values=(0, 0))
        padded_grid = np.pad(grid, [(int(abs(padding_height)/2 + 1), int(abs(padding_height)/2)), (0, 0)], 'constant',  constant_values=(0, 0))
        padded_grid_parameter = 'Im&Grid'

    elif padding_height % 2 == 1 and  padding_width % 2 == 0:
        padded_im = np.pad(resized_og, [(padding_height//2 + 1, padding_height//2), (padding_width//2 , padding_width//2)], 'constant',  constant_values=(0, 0))
    
    elif padding_height % 2 == 0 and  padding_width % 2 == 1:
        padded_im = np.pad(resized_og, [(int(padding_height/2), int(padding_height/2)), (int((padding_width/2)) + 1 , int(padding_width/2))], 'constant',  constant_values=(0, 0))
    
    elif padding_height % 2 == 0 and  padding_width % 2 == 0 and padding_width > 0 and padding_height < 0:
        # print('success padding')
        padded_im = np.pad(resized_og, [(0, 0), (int((padding_width/2)) , int(padding_width/2))], 'constant',  constant_values=(0, 0))
        padded_grid = np.pad(grid, [(int(abs(padding_height)/2), int(abs(padding_height)/2)), (0, 0)], 'constant',  constant_values=(0, 0))
        padded_grid_parameter = 'Im&Grid'
    
    elif padding_height % 2 == 0 and  padding_width % 2 == 0:
        padded_im = np.pad(resized_og, [(int(padding_height/2), int(padding_height/2)), (int((padding_width/2)), int(padding_width/2))], 'constant',  constant_values=(0, 0))
    # print('padded_im' ,np.shape(padded_im))
    # print('padded_grid' ,np.shape(padded_grid))
    
    if padded_grid_parameter == None:
        padded_grid = grid
    # print('padded_im' ,np.shape(padded_im))
    # print('padded_grid' ,np.shape(padded_grid))
    # print('padded grid parameter', padded_grid_parameter)
    return padded_im, resized_og, padded_grid, padded_grid_parameter 


def find_edge(padded_array, padded_grid_array, resized_array, original_grid_array, padded_grid_parameter):    
    
    best_padded_array = None

    fixed_padded_grid = original_grid_array

    # test=Image.fromarray(padded_array*255)
    # test.show()
        
    # test=Image.fromarray(padded_grid_array*255)
    # test.show()
    
    og_x, og_y, og_coor = cross_search(padded_array, 20, 'Image')
    print('og_x', og_x)
    print('og_y', og_y)

    grid_x, grid_y, grid_coor = cross_search(padded_grid_array, 20, 'Grid')
    print('grid_x' ,grid_x)
    print('grid_y' ,grid_y)
    
    for i in range(np.shape(og_coor)[0]):
        print('i', i)
        sub_x = og_coor[i][0] - grid_coor[i][0] 
        sub_y = og_coor[i][1] - grid_coor[i][1]

        print('subx', sub_x)
        print('suby', sub_y)

        resized_array_shape = np.shape(resized_array)
        if padded_grid_parameter == 'OnlyGrid' or padded_grid_parameter == 'Im&Grid':
            grid_shape = np.shape(original_grid_array)
        else:
            grid_shape = np.shape(padded_grid_array)

        padding_width = grid_shape[1] - resized_array_shape[1]
        padding_height = grid_shape[0] - resized_array_shape[0]

        print('padding width', padding_width)
        print('padding height', padding_height)

        if padding_height % 2 == 1 and padding_width % 2== 0:
            print('1')
            try:
                print('try')
                fixed_padded_array = np.pad(resized_array, [(int(padding_height//2 -sub_y + 1), int(padding_height//2 + sub_y)), (int(padding_width//2 - sub_x), int(padding_width//2 + sub_x))], 'constant',  constant_values=(0, 0))
            except:
                print('continue')
                continue
        
        elif padding_height % 2 == 0 and padding_width % 2== 1:
            print('2')
            try:
                fixed_padded_array = np.pad(resized_array, [(int(padding_height//2 -sub_y ), int(padding_height//2 + sub_y)), (int(padding_width//2 - sub_x + 1), int(padding_width//2 + sub_x))], 'constant',  constant_values=(0, 0))
            except:
                continue
        
        elif padding_height % 2 == 1 and padding_width % 2== 1:
            print('3')
            try:
                fixed_padded_array = np.pad(resized_array, [(int(padding_height//2 -sub_y + 1), int(padding_height//2 + sub_y)), (int(padding_width//2 - sub_x + 1), int(padding_width//2 + sub_x))], 'constant',  constant_values=(0, 0))
            except:
                continue
        
        elif padding_height % 2 == 0 and padding_width % 2 == 0:
            print('4')
            try:
                fixed_padded_array = np.pad(resized_array, [(int(padding_height//2 -sub_y ), int(padding_height//2 + sub_y)), (int(padding_width//2 - sub_x), int(padding_width//2 + sub_x))], 'constant',  constant_values=(0, 0))
            except:
                continue 
        print(np.shape(fixed_padded_array))
        print(np.shape(fixed_padded_grid))

        fixed_padded_grid = fixed_padded_grid.astype(np.uint8)
        fixed_padded_array = fixed_padded_array.astype(np.uint8)
        
        cv2.imwrite(path + '/a.png', fixed_padded_array*255)
        
        fixed_padded_array1 = fixed_padded_array *255
        fixed_padded_grid1 = fixed_padded_grid *255
        dst = cv2.addWeighted(fixed_padded_grid1 , 0.5, fixed_padded_array1 , 0.8, 0)
        
        if i == 0:
            print('in if iou loop')
            cv2.imwrite(path + f'/{i}.png', dst)
            IoU_best = iou(fixed_padded_array, fixed_padded_grid)
        
        else:
            print('in else iou loop')
            cv2.imwrite(path + f'/{i}.png', dst)
            IoU = iou(fixed_padded_array, fixed_padded_grid)
            best_padded_array = fixed_padded_array
            print('IoU inside the loop', IoU)
            if IoU > IoU_best:
                print('Best IoU inside loop', IoU_best)
                IoU_best = IoU
                best_padded_array = fixed_padded_array
        
    # test = Image.fromarray(best_padded_array*255)
    # test.show()
    # test = Image.fromarray(fixed_padded_grid*255)
    # test.show()
    return best_padded_array, fixed_padded_grid, IoU_best, og_coor, grid_coor

def iou(img1, img2):
        
    bitwiseAnd = cv2.bitwise_and(img1, img2, mask=None)
    bitwiseOr = cv2.bitwise_or(img1, img2, mask=None)
    
    sum1 = np.sum(bitwiseAnd)
    sum3 = np.sum(bitwiseOr)
    
    return sum1/sum3

def sample_pixel_ratio(image_array):
    hist_trafo = Image.fromarray(image_array)

    hist = hist_trafo.histogram()      

    div = sum(hist[:int(255/2)])/ sum(hist[int(255/2):])
    return div
#%%
#AUTOMATED PART
path = os.getcwd()
print(path)
image_dir = path + '/Batch4'
grid_dir = path + '/ideal_grids'

lw_diff_list = []

img_name_list = []
images = os.listdir(image_dir)

grids = os.listdir(grid_dir)
#Take .DS Store element out of list 
grids.pop(0) 

iou_list = []
for image in images:
    try:
        img_name, img_extension = os.path.splitext(image)
        if img_extension == '.jpg':
            for grid in grids:
                grid_name, grid_extension = os.path.splitext(grid)
                if grid_name.split('_')[-1] ==  img_name.split('_')[-1]:
                    cor_im = img_name
                    cor_grid = grid_name
                    break 

            print(img_name)
            print(grid_name)

            # im = Image.open(f'{image_dir}/{cor_im}{img_extension}', 'r').convert('L')
            im = cv2.imread(f'{image_dir}/{cor_im}{img_extension}', cv2.IMREAD_GRAYSCALE)
            grid = cv2.imread(f'{grid_dir}/{cor_grid}{grid_extension}', cv2.IMREAD_GRAYSCALE)

            #IMAGE PREPREPROCESSING
            dim = (2000, 2000)

            # im = np.array(im)
            im = cv2.resize(im, dsize = dim)

            div = sample_pixel_ratio(im)

            if div < 0.07:
                simplify_parameter = 180

            elif div > 0.15:
                simplify_parameter = 100
            
            else: 
                simplify_parameter = 120

            simplified_image = simplify(im, simplify_parameter)
            
            rediv = sample_pixel_ratio(np.array(simplified_image))
            
            if  0.1 < rediv - div:
                simplify_parameter = 140
                simplified_image = simplify(im, simplify_parameter)
                simplified_image.show()
                # rediv = sample_pixel_ratio(np.array(simplified_image))
                

            im = np.array(simplified_image)//255
            im = im.astype(int)

            im = np.where((im==0)|(im==1), im^1, im)

            # test = im.astype(np.uint8)
            # test = Image.fromarray(test*255)
            # test.show()
            #GRID PREPROCESSING
            grid = cv2.resize(grid, dim)
            grid = grid/255

            grid[grid < 0.8] = 0
            grid[grid > 0.8] = 1

            grid = grid.astype(int)

            grid = np.where((grid==0)|(grid==1), grid^1, grid)

            # grid1 = grid.astype(np.uint8)
            # grid1 = Image.fromarray(grid1*255)
            # grid1.show()

            #DENOISE
            image_sep = measure.label(im)

            for i in np.unique(image_sep, return_counts=False):
                        if np.count_nonzero(image_sep == i) < 1300:
                            im[image_sep==i] = 0

            #CROP
            offset = 40
            gshape = np.shape(im)
            if grid_name.split('_')[-1] == '01' or grid_name.split('_')[-1] == '02' or grid_name.split('_')[-1] == '03' or grid_name.split('_')[-1] == '06' or grid_name.split('_')[-1] == '09':

                for i in range(int(gshape[0]/2)):
                    for j in range(int(gshape[1]/2)):
                        if_break = False
                        if i > 45 and j > 45:
                            if im[i][j] == 1:
                                im = im.astype(np.uint8)
                                grayscale = Image.fromarray(im*255)
                                crop_im = grayscale.crop(box = (0, i-offset, gshape[1], gshape[0])) 
                                crop_im1 = np.array(crop_im)//255
                                crop_im_shape = np.shape(crop_im1)
                                if_break = True
                                break
                    if if_break == True: break

                for i in range(int(crop_im_shape[1]/3)):
                    for j in range(int(crop_im_shape[0]/3) ):
                        if_break = None
                        if i > 45 and j > 45 and crop_im1[j][i] == 1 and j <= int(crop_im_shape[1]/2) :
                            crop_im = crop_im.crop(box = (i-offset, 0, crop_im_shape[1], crop_im_shape[0]))
                            crop_im1 = np.array(crop_im)//255
                            crop_im_shape = np.shape(crop_im1)
                            if_break = True
                            break
                    if if_break == True: break

                for i in reversed(range(crop_im_shape[0]-45)):
                    for j in reversed(range(crop_im_shape[1]-45)):
                        if_break = None
                        if crop_im1[i][j] == 1 and j >= int(crop_im_shape[0]/2):
                            crop_im = crop_im.crop(box = (0, 0, crop_im_shape[1], i+offset))
                            crop_im1 = np.array(crop_im)//255
                            crop_im_shape = np.shape(crop_im1)
                            if_break = True
                            break
                    if if_break == True: break

                for i in reversed(range(crop_im_shape[1]-45)):
                    counter = 0
                    for j in reversed(range(crop_im_shape[0]-45)) :
                        if_break = None
                        diff = abs(crop_im1[j][i] -  crop_im1[j-1][i])
                        if diff== 1 :
                            counter += 1
                            if counter > 20:
                                crop_im = crop_im.crop(box = (0, 0, i+offset, crop_im_shape[0])) 
                                if_break = True
                                break
                    if if_break == True: break

            elif grid_name.split('_')[-1] == '04' or grid_name.split('_')[-1] == '05' or grid_name.split('_')[-1] == '07' or grid_name.split('_')[-1] == '08': 
                for i in range(int(gshape[0]/2 + 10)):
                    for j in range(int(gshape[1]/2)):
                            if_break = False    
                            if i > 45 and j > 70 and im[i][j] == 1:
                                im = im.astype(np.uint8)
                                grayscale = Image.fromarray(im*255)
                                crop_im = grayscale.crop(box = (0, i-offset, gshape[1], gshape[0])) 
                                crop_im_arr = np.array(crop_im)//255
                                crop_im_shape = np.shape(crop_im_arr)
                                if_break = True
                                break
                    if if_break == True: break

                for i in range(crop_im_shape[1]):
                    for j in range(crop_im_shape[0]):
                        if_break = None
                        if i > 70 and j > 70 and crop_im_arr[j][i] == 1 and j <= int(crop_im_shape[1]/2):
                            crop_im = crop_im.crop(box = (i-offset, 0, crop_im_shape[1], crop_im_shape[0]))
                            crop_im_arr = np.array(crop_im)//255
                            crop_im_shape = np.shape(crop_im_arr)
                            if_break = True
                            break
                    if if_break == True: break

                for i in reversed(range(int(crop_im_shape[1] - 50))):
                    for j in range(int(crop_im_shape[0]/2)):
                        if_break = None
                        if crop_im_arr[j][i] == 1:
                            crop_im = crop_im.crop(box = (0, 0, i+offset, crop_im_shape[0])) 
                            crop_im_arr = np.array(crop_im)//255
                            crop_im_arr = crop_im_arr.astype(int)
                            crop_im_shape = np.shape(crop_im_arr)
                            if_break = True
                            break
                    if if_break == True: break


                for i in reversed(range(crop_im_shape[0]-50)):
                    counter = 0
                    for j in range(int(crop_im_shape[1]/2)):
                        if_break = None
                        diff = abs(crop_im_arr[i][j] -  crop_im_arr[i][j-1])
                        if diff == 1:
                            counter += 1
                            if counter > 40:
                                crop_im = crop_im.crop(box = (0, 0, crop_im_shape[1], i+offset)) 
                                if_break = True
                                break
                    if if_break == True: break

            # crop_im.show()

            cropped_image = np.array(crop_im)/255
            cropped_image = cropped_image.astype(int)

            a, b, c ,d= image_resize(cropped_image, grid, lw_diff_list)
            h, new_grid, IoU, final_padded_array, _ = find_edge(a, c, b, grid, d)

            # h = np.array(h)
            h = h *255
            # h = h.astype(np.uint8)

            new_grid = new_grid *255
            # new_grid = new_grid.astype(np.uint8)

            print('Image shape', np.shape(h))
            print('Image type', type(h))
            print('Grid shape', np.shape(new_grid))
            print('Grid type', type(new_grid))
            # IoU = iou(h, new_grid)
            print('IoU:', IoU)
            dst = cv2.addWeighted(new_grid , 0.5, h , 0.8, 0)
            cv2.imwrite(path + f'/IoU/{img_name}_IoU.png', dst)
            cv2.imwrite(path + f'/padded_images/{img_name}_padded.png', a*255)
            cv2.imwrite(path + f'/final_padding/{img_name}_final_padding.png', h)
            # cv2.imshow('Blended Image',dst)
            iou_for_list = f'{img_name} :{IoU}'
            iou_list.append(iou_for_list)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    except:
        print('iteration failed')
        continue
lw_diff_list = np.array(lw_diff_list)
lw_mean = np.mean(lw_diff_list)
lw_std = np.std(lw_diff_list)
print('lw_diff_list', lw_diff_list)
print('lw_mean', lw_mean)
print('lw_std', lw_std)
# file= open('BATCH4_redo.txt','w')
# for item in iou_list:
#     file.write(item + "\n")
# file.close()
# %%
