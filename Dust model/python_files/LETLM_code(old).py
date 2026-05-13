import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import torch
from tqdm import tqdm
import math
from python_files import base_functions as bf
from python_files import TLM_tests as TT



# I am now going to try make the LETLM of the Lorenz 96 Model that is timestepped with RK4

# We we are going to start by generating the ensemble members
def ensemble_generator(ensemble_size, model_parameters, model, perturbation_size, spin_up_time):
    # we first extract the model variables
    (N, dx, dt, alpha, beta, F_L96) = model_parameters
    # we now run the model to spin up the model
    x_in = torch.rand((N,1))
    results = x_in
    for i in range(spin_up_time):
        x_out = model(x_in, model_parameters)
        results = torch.cat((results, x_out), dim = 1)
        x_in = x_out

    # We will create the tensors to hold the ensembles
    number_of_variables =  N # n = number of varables at each point and N = number of points
    Xi = torch.zeros(number_of_variables, ensemble_size)
    X = torch.zeros(number_of_variables, ensemble_size)
    Chi = torch.zeros(number_of_variables, ensemble_size)

    for i in range(ensemble_size):
        x_pert = torch.randn(number_of_variables,1) * perturbation_size
        x_in_pert = x_out + x_pert
        x_out_pert = model(x_in_pert,model_parameters)
        x_out_unpert = model(x_out, model_parameters)
        Xi_i = x_out_pert - x_out_unpert
        X[:,i] = x_pert.squeeze()
        Xi[:,i] = Xi_i.squeeze()
        x_out_pert = model(x_in_pert,model_parameters)
        x_out_unpert = model(x_out, model_parameters)
        Chi_i = x_out_pert - x_out_unpert
        Chi[:,i] = Chi_i.squeeze()
    return X, Xi, Chi


# We will now define a function that selects the 9 members of each row 
def member_past_selector(grid_index,X,N,ensemble_size):
    subset = torch.zeros(9,ensemble_size)
    for i in range(9):
        j = (grid_index +i - 4)%N
        # print(grid_index, j)
        subset[i,:] = X[j,:]
    return subset
    



def member_current_selector(grid_index,Xi,N,ensemble_size):
    subset = torch.zeros(1,ensemble_size)
    j = grid_index
    subset[0,:] = Xi[j,:]
    # print(subset.shape)
    return subset



def little_LETLM(current_subset, past_subset):
    etlm = current_subset @ past_subset.T@(torch.inverse(past_subset@ past_subset.T))
    return etlm


# We will now define a function that puts M_i into M 
def put_in_place(M_i, N, M, i):
    M_i = M_i.squeeze()
    # For debugging use this test tensor
    # M_i = torch.tensor([1,2,3,4,5,6,7,8,9])

    M_i_reversed = torch.flip(M_i, dims=[0])

    for k in range(9):
        j = k - 4 + i
        M[j % N, i] = M_i_reversed[k]

    return M

def LETLM_generator(ensemble_size, model_parameters, model, perturbation_size, spin_up_time):
    (N, dx, dt, alpha, beta, F_L96) = model_parameters
    # We start by generating the ensemble members
    X, Xi, Chi = ensemble_generator(ensemble_size, model_parameters, model, perturbation_size, spin_up_time)
    # We now create an empty tensor that will become our M matrix
    M = torch.zeros(N,N)
    # We now need to calculate the LETLM for each variable in the model 
    for i in tqdm(range(N)):
        # The first step is to select the required ensemble members:
        # We start with the past time step 
        X_i = member_past_selector(i,X,N,ensemble_size)
        # Now the current time step
        Xi_i = member_current_selector(i,Xi,N,ensemble_size)
        # print(Xi_i.shape)
        
        # We now calculate the local LETLM
        M_i = little_LETLM(Xi_i, X_i)
        # print(M_i.shape)

        # We now need to put M_i into the correct position in M
        M = put_in_place(M_i, N, M, i)
    return M
        


        
        