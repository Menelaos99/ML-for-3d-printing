import numpy as np 
import pandas as pd

annika_parameters = pd.read_excel('ML_Prusa_FiberAjustment_Setup.xlsx')
annika_parameters= annika_parameters.fillna(0)
annika_parameters = annika_parameters.to_numpy()

iou_df = pd.read_csv('ious.txt', sep=":", header=None).to_numpy()

speed = []
divisor = []
hv= []
temp = []
humidity = []
iou_array = []
for i in range(np.shape(iou_df)[0]):
    iou = iou_df[i][0]
    find_grid = iou_df[i][0].split('_')[-1]
    find_column = 42 + int(find_grid)
    # print('next')
    # print(iou_df[i][0])
    # print(find_column)
    for i in range(np.shape(annika_parameters)[0]):
        if str(iou).replace(" ", "") == str(annika_parameters[i][find_column]).replace(" ", ""):
            iou_array.append(iou_df[i,1])
            speed.append(annika_parameters[i][30])
            divisor.append(annika_parameters[i][28])
            hv.append(annika_parameters[i][20])
            humidity.append(annika_parameters[i][14])
            temp.append(annika_parameters[i][19])

speed = np.asarray(speed).reshape(np.shape(speed)[0], 1)
divisor = np.asarray(divisor).reshape(np.shape(divisor)[0], 1)
hv = np.asarray(hv).reshape(np.shape(hv)[0], 1)
humidity = np.asarray(humidity).reshape(np.shape(humidity)[0], 1)
temp = np.asarray(temp).reshape(np.shape(temp)[0], 1)

gpbo_parameters = np.concatenate([temp, humidity,speed, divisor, hv], axis = 1, dtype=np.float)
iou_array = np.array(iou_array)
print(gpbo_parameters.shape)
print(iou_array.shape)
