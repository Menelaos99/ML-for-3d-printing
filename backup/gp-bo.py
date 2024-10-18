
#%%
import pandas as pd 
import numpy as np 
# import GPy
import matplotlib.pyplot as plt

from scipy.stats import norm

# %%
import pandas as pd
import numpy as np
import os
from itertools import chain
import math
import tqdm
import torch
import gpytorch
seed = 42
normalization_flag = False
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

gpbo_parameters = np.concatenate([temp, humidity,speed, divisor, hv], axis = 1, dtype=np.float)
iou_array = np.array(iou_array)


u, i  = np.unique(gpbo_parameters, axis=0, return_index=True)
iou_array = iou_array[i]
gpbo_parameters = gpbo_parameters[i]





IoU = np.array(iou_array)
# IoU = IoU[:, 1].reshape(len(IoU), 1)
print(IoU.shape)
if inversion_flag :
    IoU = 1/IoU
# print(IoU)
# print('a')
if normalization_flag :
    gpbo_parameters, param_max, param_min = normalization(gpbo_parameters)
    IoU ,iou_max, iou_min = normalization(IoU)



#%%

data_dim = 5
class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()

            # self.covar_module =  gpytorch.kernels.ArcKernel(gpytorch.kernels.MaternKernel(nu=1.5),
            #                     angle_prior=gpytorch.priors.GammaPrior(0.5,1),
            #                     radius_prior=gpytorch.priors.GammaPrior(3,1),
            #                     ard_num_dims=4)
            # self.covar_module =gpytorch.kernels.ScaleKernel(\
            self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=5) + gpytorch.kernels.LinearKernel(num_dimensions = 5)
            # self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims= 5) +  gpytorch.kernels.LinearKernel(num_dimensions = 5)
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        

def train(optimizer, model,mll, train_x, train_y):
    for i in range(5000):

        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
IoU = np.reshape(IoU, (IoU.shape[0]))

likelihood = gpytorch.likelihoods.GaussianLikelihood()
# likelihood  = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.ones(IoU.shape[0]) *10, learn_additional_noise=True)
parameters_normalized = torch.from_numpy(gpbo_parameters.astype(np.float32))
IoU = torch.from_numpy(IoU.astype(np.float32))
model = GPRegressionModel( parameters_normalized, IoU, likelihood)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
# mll = gpytorch.mlls.NoiseModelAddedLossTerm(likelihood, model)

model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.covar_module.parameters()},
    {'params': model.mean_module.parameters()},
    {'params': model.likelihood.parameters()},
], lr=0.0001)


train(optimizer, model,mll, parameters_normalized, IoU)

model.eval()
# %%

# %%
def find_optimum(observations, model):
    best_so_far = 0

    for i in range( observations.shape[0]):
            pred_temp =  model(observations[i:i+1]).mean.item()
            
            # pred = reverse_normalization(pred_temp, iou_max, iou_min)
            # pred = 1/pred
            if pred_temp > best_so_far:
                best_so_far = pred_temp
    return best_so_far
# %%
print(find_optimum(parameters_normalized, model))
# %%


def expected_improvement(mu, std, optimal_so_far, gpr, tradeoff=0.01):
    # sigma = sigma.reshape(-1, 1)
    mu_sample_opt = optimal_so_far
    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - tradeoff
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std == 0.0] = 0.0
    return ei


def get_score(Xsamples,optimal_so_far, model,method = "pe",tradeoff = 0.5 ):
    likelihood.eval()
    model.eval()
    mu, std = likelihood(model(Xsamples)).mean.detach().numpy(), np.sqrt(likelihood(model(Xsamples)).variance.detach().numpy())
    print(std.min(), std.max())
    # mu = mu[:, 0]
    # std = std[:,0]
    plt.hist(std)
    plt.show()
    std = np.reshape(std, (std.shape[0],1))
    mu = np.reshape(mu, (mu.shape[0],1))
    if method == "pi":
        
        scores = norm.cdf((mu - optimal_so_far) / (std+1E-9))  
    elif method == "pe":
        scores = expected_improvement(mu, std,optimal_so_far, model, tradeoff) 
        
    elif method == "ucb":
        scores = mu + tradeoff*std
    return scores

def get_suggestions(optimal_so_far, model,method = "pe",humidity_value = 21,temperature_value = 25, tradeoff = 0.5):
    Xsamples = []
    sample_size = 100000
    humidity_value
    # temp, humidity,speed, divisor, hv],
    Xsamples = np.random.uniform(size=(sample_size,5), low = 0, high =1)
    if normalization_flag:
        Xsamples[:,0] = normalization_with_inputs(temperature_value, param_max[0],param_min[0])
        Xsamples[:,1] = normalization_with_inputs(humidity_value, param_max[1],param_min[1])
        Xsamples[:,2] = normalization_with_inputs(Xsamples[:,2]*600+200, param_max[2],param_min[2])
        Xsamples[:,3] = normalization_with_inputs(Xsamples[:,3]*950 + 50, param_max[3],param_min[3])
        Xsamples[:,4] = normalization_with_inputs(Xsamples[:,4]*2+3.5, param_max[4],param_min[4])
    else:
        Xsamples[:,0] = temperature_value
        Xsamples[:,1] = humidity_value
        Xsamples[:,2] = Xsamples[:,2]*600+200
        Xsamples[:,3] = Xsamples[:,3]*950 + 50
        Xsamples[:,4] = Xsamples[:,4]*2+3.5


    Xsamples = torch.from_numpy(Xsamples.astype(np.float32))
    scores = get_score(Xsamples,optimal_so_far, model,method, tradeoff)
    ix = np.argmax(scores) #maximisation


    return Xsamples[ix]
# %%
acquisition_values = []
humidity_value = 43
temperature_value = 23.4  
# tradeoff = 0.01  # higher tradeoff = more exploration 
optimal_so_far = find_optimum(parameters_normalized, model)
print(optimal_so_far)
for tradeoff in [0.5]:
    for method in ["pi"]:
        best = get_suggestions(optimal_so_far,model,method,humidity_value, temperature_value, tradeoff)
        best = np.reshape(best, (1,best.shape[0]))

        results = model(best).mean.detach().numpy()
        results_variance = model(best).variance.detach().numpy()
        if normalization_flag:
            results = reverse_normalization(results, iou_max, iou_min)
        # results = 1/results
            results_variance = reverse_normalization(results_variance, iou_max, iou_min)
        # results_variance = 1/results_variance
            best = reverse_normalization(np.array(best),param_max, param_min ) 
        print(tradeoff, method)
        print(results[0], results_variance[0])
        print(' temp,        humidity,    speed,        divisor,       hv')
        print(best[0])
        print()
# %%
# scores = get_score(Xsamples,optimal_so_far, model,method, tradeoff)



# %%
