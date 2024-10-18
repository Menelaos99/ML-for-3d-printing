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
#%%
df_parameters_old = pd.read_csv( 'correctparameters.csv', sep=r';', engine='python')
df_parameters_old= df_parameters_old.fillna(0)
df_parameters_old = df_parameters_old.to_numpy()
#%%


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


#%%
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

gpbo_parameters = np.concatenate([humidity,temp, divisor, speed, hv], axis = 1, dtype=np.float)


iou_array = np.array(iou_array)
iou_array = np.concatenate([np.reshape(iou_array,(iou_array.shape[0],1) ), np.reshape(IoU_old, (IoU_old.shape[0],1))], axis = 0)
gpbo_parameters = np.concatenate([np.array(gpbo_parameters), parameters_ges_old], axis = 0).astype(None)

print(gpbo_parameters.shape)
print(iou_array)

#%%
parameters_ges = gpbo_parameters
u, i  = np.unique(gpbo_parameters, axis=0, return_index=True)

#%%
print(iou_array.shape , "before")
u, i  = np.unique(parameters_ges, axis=0, return_index=True)
iou_array = iou_array[i]
parameters_ges = parameters_ges[i]
print(parameters_ges)

print(iou_array.shape , "after")


IoU = np.array(iou_array)
IoU = np.reshape(IoU, (IoU.shape[0],1))
if inversion_flag :
    IoU = 1/IoU

# print(IoU)
# print('a')
if normalization_flag :
    parameters_normalized, param_max, param_min = normalization(parameters_ges)
    iou_normalized ,iou_max, iou_min = normalization(IoU)

# normalization_flag = False
# parameters_normalized = parameters_ges
# iou_normalized = IoU
# print(iou_normalized.shape, IoU.shape)
#%%

from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from matplotlib import pyplot as plt
# %%
data_dim = 5
class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(LargeFeatureExtractor, self).__init__()

        self.add_module('linear3', torch.nn.Linear(data_dim, 5))
        self.add_module('relu3', torch.nn.ReLU())
        # self.add_module("bn2",torch.nn.BatchNorm1d(4))
        # self.add_module('linear4', torch.nn.Linear(4, 3))
        # self.add_module("bn1",torch.nn.BatchNorm1d(3))
        # self.add_module('tan', torch.nn.ReLU())

feature_extractor = LargeFeatureExtractor()

class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            # self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            #     gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
            #     num_dims=2, grid_size=100
            # )
            # self.covar_module =  gpytorch.kernels.ArcKernel(gpytorch.kernels.MaternKernel(nu=1.5),
            #                     angle_prior=gpytorch.priors.GammaPrior(0.5,1),
            #                     radius_prior=gpytorch.priors.GammaPrior(3,1),
            #                     ard_num_dims=data_dim)
            # self.covar_module =gpytorch.kernels.ScaleKernel(\
                # gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=5))
            self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=5) + gpytorch.kernels.LinearKernel(num_dimensions = 5)

            # self.covar_module =gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=5)

            self.feature_extractor = feature_extractor

            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(0., 1.)

        def forward(self, x):
            # We're first putting our data through a deep net (feature extractor)
            # x = self.feature_extractor(x)
            # x = self.scale_to_bounds(x)  # Make the NN values "nice"

            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        

def train(optimizer, model,mll, train_x, train_y):
    # iterator = tqdm.notebook.tqdm(range(1000))
    for i in range(5000):
        # print(iterator)
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        # Calc loss and backprop derivativesGitHub
        optimizer.step()


# %%
prediction = []
truth = []

# normalization_flag = True
std = []
for i in range(IoU.shape[0]):
    print(i)
    train_x = parameters_normalized[np.arange(parameters_normalized.shape[0])!=i]
    test_x = parameters_normalized[i]
    train_y = iou_normalized[np.arange(IoU.shape[0])!=i]
    test_y = iou_normalized[i]

    train_y = np.reshape(train_y, (train_y.shape[0]))
    test_y = np.reshape(test_y, (test_y.shape[0]))
    test_x = np.reshape(test_x, (1,test_x.shape[0]))

    train_x = torch.from_numpy(train_x.astype(np.float32))
    train_y = torch.from_numpy(train_y.astype(np.float32))
    test_x = torch.from_numpy(test_x.astype(np.float32))
    test_y = torch.from_numpy(test_y.astype(np.float32))



    # print(y_test)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(train_x, train_y, likelihood)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # if torch.cuda.is_available():
    #     model = model.cuda()
    #     likelihood = likelihood.cuda()
    training_iterations = 1000
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        # {'params': model.feature_extractor.parameters()},
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': model.likelihood.parameters()},
    ], lr=0.0001)


    train(optimizer, model,mll, train_x, train_y)


    model.eval()
    likelihood.eval()
    with torch.no_grad():
        preds = model(test_x)
    
    test_y = np.array(test_y.item())
    mean = np.array(preds.mean.item())
    variance = np.array(preds.variance.item())
    print(variance,test_y,mean)
    if normalization_flag:
        mean = reverse_normalization(mean, iou_max, iou_min)
        variance = (reverse_normalization(variance, iou_max, iou_min))
        test_y = reverse_normalization(test_y, iou_max, iou_min)
    else:
        mean = mean
        variance = variance
        test_y = test_y
    mean = abs(mean)
    prediction.append(mean)
    std.append(variance)
    truth.append(test_y)
prediction = np.array(prediction)
std = np.array(std)
truth = np.array(truth)
print(std)
# %%
prediction =np.reshape(prediction, (prediction.shape[0],1))
import matplotlib.pyplot as plt
plt.plot(prediction, truth, 'o')
print("relative error  = ",100*np.mean(abs(1/prediction-1/truth)/(1/truth)))
plt.show()
# %%
# %%
if inversion_flag:
    pred_cor = 1/prediction
    truth_cor = 1/truth
    std_cor = np.sqrt(1/std.astype(float))

else:
    pred_cor = prediction
    truth_cor = truth
    std_cor = std.astype(float)
error = 100*np.mean(abs(pred_cor - truth_cor)/truth_cor)

pred_cor = np.reshape(pred_cor,pred_cor.shape[0])
truth_cor = np.reshape(truth_cor,truth_cor.shape[0])
std_cor = np.reshape(std_cor,std_cor.shape[0])
plt.plot(pred_cor, truth_cor, 'o')
# plt.errorbar(pred_cor, truth_cor, std_cor ,fmt = 'o', label='ERROR = ' + str(np.round(error,2)) + ' %')
plt.title("Predictions with error bars")
plt.xlabel('predicted IoU')
plt.ylabel('true IoU')
plt.legend(loc='upper left',bbox_to_anchor=(0.1, 1),markerscale=0,handletextpad=-2.0, handlelength=0,frameon=False,numpoints=1)
# plt.savefig("NO_DKL_error_predcition_normalizattion_AND_inversion.pdf", dpi = 1000)
#%%
plt.hist(abs(100*(pred_cor-truth_cor)/truth_cor))
plt.xlabel('% error')
plt.ylabel('occurencies')
# plt.savefig("NO_DKL_error_hist_norm_AND_inv.pdf", dpi = 1000)

# %%
# %%
# %%
