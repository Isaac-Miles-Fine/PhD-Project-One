import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import torch
from tqdm import tqdm
import math
from python_files import base_functions as bf


# We will start by defining our error function
def norm_error(x_pert, N, model,dt, M):
    x_in_pert = x_out + x_pert
    x_out_pert = model(x_in_pert, dt)
    x_out_unpert = model(x_out, dt)
    diff = x_out_pert - x_out_unpert - torch.matmul(M, x_pert)
    return diff


def normalisor(sd, N, model,dt):
    x_in_pert = x_out + x_pert
    x_out_pert = model(x_in_pert, dt)
    x_out_unpert = model(x_out, dt)
    norm = x_out_pert - x_out_unpert
    return norm


# We will now define a very quick generic TLM test that will be used for the multistep TLM test 
def generic_TLM_test(diff, norm):
    error = math.sqrt(torch.matmul(diff.T, diff)/torch.matmul(norm.T, norm))
    return error
    


def TLM_test(sd, N, model,dt, n_trials, M):
    TLM_error = 0
    for i in range(n_trials):
        torch.manual_seed((42*i)**4)
        x_pert = torch.randn(2*N,1)*sd
        # print(x_pert[4])
        diff = norm_error(x_pert, N, model,dt, M)
        norm = normalisor(x_pert, N, model,dt)
        # ith_error = math.sqrt(torch.matmul(diff.T, diff)/torch.matmul(norm.T, norm))
        ith_error = generic_TLM_test(diff, norm)
        TLM_error += ith_error
    return TLM_error/n_trials




    

