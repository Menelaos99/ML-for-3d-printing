#%%
import pandas as pd
import numpy as np
import os
from itertools import chain
import math
import tqdm
import torch
import gpytorch

seed = 42
normalization_flag = True
inversion_flag = False
torch.manual_seed(seed)
np.random.seed(seed)

def normalization(a):
    return (a - a.min(axis = 0))/(a.max(axis = 0) - a.min(axis = 0)), a.max(axis = 0), a.min(axis = 0)

def normalization_with_inputs(a, amax, amin):
    return (a - amin)/(amax - amin)

def reverse_normalization(a,amax,amin):
    return a*(amax-amin) + amin

def data_processing(df, name):
    temp = df[name]
    temp = np.array(temp)
    temp = np.reshape(temp, (temp.shape[0],1))
    return temp

def flat(list_2D):
    flatten_list = list(chain.from_iterable(list_2D))
    flatten_list = np.array(flatten_list)
    return flatten_list
def parameter_extraction_function():
        df_parameters_old = pd.read_csv( 'correctparameters.csv', sep=r';', engine='python')
        df_parameters_old= df_parameters_old.fillna(0)
        df_parameters_old = df_parameters_old.to_numpy()
        df_IoU = pd.read_csv('old_IoUs.csv', sep=r';', engine='python').to_numpy()
        parameters = []
        speed = []
        for i in range(len(df_IoU)):
            IoU_name = df_IoU[i][0].split('.')
            for j in range(len(df_parameters_old)):
                parameters_name = df_parameters_old[j][6]
                if IoU_name[0] == parameters_name:
                    parameters.append(np.array(df_parameters_old[j][:3]))
                    speed.append(df_parameters_old[j][3])




            for k in range(len(df_parameters_old)):  
                parameters_name = df_parameters_old[k][7]
                if IoU_name[0] == parameters_name:
                    parameters.append(np.array(df_parameters_old[k][:3]))
                    speed.append(df_parameters_old[k][4])



            for h in range(len(df_parameters_old)):  
                parameters_name = df_parameters_old[h][8]
                if IoU_name[0] == parameters_name:
                    parameters.append(np.array(df_parameters_old[h][:3]))
                    speed.append(df_parameters_old[h][5])
        speed = np.asarray(speed)
        speed[speed == 0.0 ] = 250

        parameters = np.concatenate(parameters, axis=0)
        parameters = parameters.reshape(28,3)

        speed = speed.reshape(28,1)
        parameters_ges_old =  np.concatenate([parameters, speed, np.ones((28,1))*(-4.5)], axis=1)
        parameters_ges_old = np.array(parameters_ges_old)
        IoU = np.array(df_IoU)
        IoU_old = IoU[:, 1].reshape(len(IoU), 1)



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

        gpbo_parameters = np.concatenate([humidity,temp, divisor, speed, hv], axis = 1, dtype=np.float)

        # print(hv, "VOLTAGE")
        iou_array = np.array(iou_array)
        # iou_array = np.concatenate([np.reshape(iou_array,(iou_array.shape[0],1) ), np.reshape(IoU_old, (IoU_old.shape[0],1))], axis = 0)
        # gpbo_parameters = np.concatenate([np.array(gpbo_parameters), parameters_ges_old], axis = 0).astype(None)

        # print(gpbo_parameters.shape)
        # print(iou_array)


        parameters_ges = gpbo_parameters
        u, i  = np.unique(gpbo_parameters, axis=0, return_index=True)

        # print(iou_array.shape , "before")
        u, i  = np.unique(parameters_ges, axis=0, return_index=True)
        iou_array = iou_array[i]
        parameters_ges = parameters_ges[i]
        # print(parameters_ges)
        return iou_array , parameters_ges



