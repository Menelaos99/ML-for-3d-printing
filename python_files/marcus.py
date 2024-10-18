from PIL import Image, ImageOps 
import numpy as np
from skimage.filters.rank import median
from skimage.morphology import disk
from numpy import linalg as ln
import cv2
from matplotlib import pyplot
from skimage import measure
import scipy
from scipy import ndimage

img_or = cv2.imread('/Users/menelaos/Desktop/3d_printing_project/IoU/PCL_MR34_x100_FMEW_grid_01_IoU.jpg')
grid = cv2.imread('/Users/menelaos/LRZ Sync+Share/FMEWgoesML (Annika Hangleiter)/ideal_grids/Grid_01.png', cv2.IMREAD_GRAYSCALE)

img = cv2.cvtColor(img_or, cv2.COLOR_BGR2GRAY)

ver_pixel_coordinates = []
hor_pixel_coordinates = []

dim = (2000, 2000)
grid = cv2.resize(grid, dim)

grid[grid<200] = 0
grid[grid>200] = 255

# test = Image.fromarray(grid)
# test.show()
sobel_dx = scipy.ndimage.sobel(grid, 0)
sobel_dy = scipy.ndimage.sobel(grid)

sobel_dx[sobel_dx>100] = 255
sobel_dx[sobel_dx<100] = 0

sobel_dy[sobel_dy>100] = 255
sobel_dy[sobel_dy<100] = 0

# test = Image.fromarray(sobel_dy)
# test.show()

image_sep = measure.label(sobel_dx)
for i in np.unique(image_sep, return_counts=False):
    if np.count_nonzero(image_sep == i) < 1000:
        sobel_dx[image_sep==i] = 0
image_sep = measure.label(sobel_dx)

unique_el = np.unique(image_sep, return_index=True)
indexes = np.unravel_index(unique_el[1][0:], (2000, 2000))

counter = 0
for x, y in zip(indexes[0][1::2], indexes[1][1::2]):
    counter += 1
    if counter == 2:
        i_temp3 = x
        j_temp3 = y
        print('itemp3', i_temp3)
        print('jtemp3', j_temp3)
# test = Image.fromarray(sobel_dx)
# test.show()

image_sep = measure.label(sobel_dy)
for i in np.unique(image_sep, return_counts=False):
    if np.count_nonzero(image_sep == i) < 200:
        sobel_dy[image_sep==i] = 0


img[img<30] = 0
img[img>30] = 255
 
img = img/255

vert_distance_list = []
hor_distance_list = []

sobel_dy = sobel_dy/255
sobel_dx = sobel_dx/255
counter = 0
for i in range(sobel_dy.shape[1]):
    diff = abs(sobel_dy[int(sobel_dy.shape[0]/2)][i] - sobel_dy[int(sobel_dy.shape[0]/2)][i-1])
    # print('diff', diff)
    if diff == 1:
        counter += 1
        # print('counter', counter)
        if counter == 3:
            # print('x',i)
            # print('y',int(sobel_dy.shape[0]/2))
            # print('in')
            i_temp = i
            
        elif counter == 6: 
            # print('x',i)        
            # print('y',int(sobel_dy.shape[0]/2)) 
            # print('in1')
            i_temp2 = i
            distance = i_temp2 - i_temp
            break

for i in range(sobel_dy.shape[0]):
    for j in range(sobel_dy.shape[1]):
        if_break = None
        if sobel_dy[i][j] == 1:
            i_temp1 = i
            j_temp1 = j
            print('itemp1', i_temp1)
            print('jtemp1', j_temp1)
            if_break = True
            break
    if if_break == True:
        break        

for i in reversed(range(sobel_dy.shape[0])):
    for j in range(sobel_dy.shape[1]):
        if_break = None
        if sobel_dy[i][j] == 1 and j < j_temp1:
            i_temp2 = i
            j_temp2 = j
            print('itemp2', i_temp2)
            print('jtemp2', j_temp2)
            if_break = True
            break
    if if_break == True:
        break        
# for i in range(img.shape[0]):
#     counter = 0
#     for j in range(img.shape[1]):
#         if diff > 0.3:
#             counter += 1
#             if counter == 3:
#                 # print('i', i)
#                 # print('j', j)
#                 i_temp = i
#                 # print(i_temp)
img1 = img

#VERTICAL
for k in range(17):
    counter = 0
    for h in range(i_temp1 + 1, img.shape[0]):
        # print('h', h)
        diff1 =  abs(img[h][j_temp1 + int(distance/2) + (distance*2 - 5)*k] -  img[h+1][j_temp1 + int(distance/2) + (distance*2 - 5)*k])
        # print('diff', diff)
        if diff1  == 1:
            counter += 1
            # print('h', h)
            if counter > 2:
                vert_distance = h - i_temp1
                vert_distance_list.append(vert_distance)
                ver_pixel_coordinates.append((j_temp1 + int(distance/2) + (distance*2 - 5)*k, i_temp1 + 1))
                ver_pixel_coordinates.append((j_temp1 + int(distance/2) + (distance*2 - 5)*k, h))
                break

for k in range(17):
    counter = 0
    for h in reversed(range(0, i_temp2 -2)):
        # print('h', h)
        diff1 =  abs(img[h][j_temp2 + int(distance/2) + (distance*2 - 5)*k] -  img[h-1][j_temp2 + int(distance/2) + (distance*2 - 5)*k])
        # print('diff', diff)
        if diff1  == 1:
            counter += 1
            # print('h', h)
            if counter > 1:
                vert_distance = h - i_temp2
                vert_distance_list.append(vert_distance)
                ver_pixel_coordinates.append((j_temp2 + int(distance/2) + (distance*2 - 5)*k, i_temp2 + 1))
                ver_pixel_coordinates.append((j_temp2 + int(distance/2) + (distance*2 - 5)*k, h))
                break

#HORIZONTAL
for k in range(17):
    counter = 0
    for h in range(j_temp3, img.shape[1]):
        # print('h', h)
        diff1 =  abs(img[i_temp3 + int(distance/2) + (distance*2 -5)*k][h] - img[i_temp3 + int(distance/2) + (distance*2 -5)*k][h+1])
        # print('diff', diff)
        if diff1  == 1:
            counter += 1
            # print('h', h)
            if counter > 2:
                hor_distance = h - j_temp3
                hor_distance_list.append(hor_distance)
                hor_pixel_coordinates.append((h, i_temp3 + int(distance/2) + (distance*2 -5)*k))
                hor_pixel_coordinates.append((j_temp3 + 1, i_temp3 + int(distance/2-5) + (distance*2 )*k-5))
                break


for k in range(17):
    for h in reversed(range(i_temp1, img.shape[0])):
        img1[h][j_temp1 + int(distance/2) + (distance*2 - 6)*k] = 1

for k in range(17):
    for h in range(i_temp3, img.shape[1]):
        img[i_temp3 + int(distance/2) + (distance*2 - 5)*k][h] = 1

########## VISUALIVATION ##########
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (0, 0, 255) 
font_thickness = 2

for point1, point2, text in zip(ver_pixel_coordinates[0::2], ver_pixel_coordinates[1::2], vert_distance_list): 
    cv2.line(img_or, point1, point2, [0, 0, 255], 2)
    text  = f'{str(text)}px'
    cv2.putText(img_or, text, (point1[0]+2, point1[1] + int((point2[1]-point1[1])/2)), font, font_scale, font_color, font_thickness)

for point1, point2, text in zip(hor_pixel_coordinates[0::2], hor_pixel_coordinates[1::2], hor_distance_list): 
    print('point1', point1)
    print('point2', point2)
    cv2.line(img_or, point1, point2, [0, 0, 255], 2)
    text  = f'{str(text)}px'
    cv2.putText(img_or, text, (point2[0]+2, point2[1] + int((point1[1]-point2[1])/2)), font, font_scale, font_color, font_thickness)


test = Image.fromarray(img_or)
test.show()

# test = Image.fromarray(img1*255)
# test.show()
print("ver_pixel_coordinates:", ver_pixel_coordinates)
print("hor_pixel_coordinates:", hor_pixel_coordinates)
cv2.imwrite('/Users/menelaos/Desktop/3d_printing_project/return_measue.jpg', img_or)

