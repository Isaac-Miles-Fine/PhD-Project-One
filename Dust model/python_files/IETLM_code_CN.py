import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import torch
from tqdm import tqdm
import math
from python_files import base_functions as bf
from python_files import TLM_tests as TT



def spin_up(x, model_parameters, model,spin_up_time):
    (N, dx, dt, alpha, beta, F_L96) = model_parameters 
    x_in = x

    for i in range(spin_up_time):
        x_out = model(x_in, model_parameters)
        # print(x_out.shape)
        x_in = x_out
    return x_out




# We will now define a function that selects the 9 members of each row 
def member_past_selector(grid_index,X,N,ensemble_size):
    subset = torch.zeros(5,ensemble_size)
    for i in range(5):
        j = (grid_index +i - 2)%N
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

def put_in_place(M_i, N, M, i):
    M_i = M_i.squeeze()
    # For debugging use this test tensor
    # M_i = torch.tensor([1,2,3,4,5,6,7,8,9])

    M_i_reversed = torch.flip(M_i, dims=[0])

    for k in range(5):
        j = k - 2 + i
        M[j % N, i] = M_i_reversed[k]

    return M

# def put_in_place(M_i, N, M, i):
#     # Ensure M_i is a 1D vector even if it's a scalar
#     M_i = M_i.view(-1) 
#     num_elements = len(M_i)
    
#     if num_elements == 1:
#         # This is for the N_tilde diagonal component
#         M[i, i] = M_i[0]
#     else:
#         # This is for the L_tilde 13-point stencil component
#         for k in range(num_elements):
#             input_idx = (i + k - 8) % N
#             M[i, input_idx] = M_i[k]
#     return M




def IETLM_generator(ensemble_size, sd, x_out, model_parameters, model):
    (N, dx, dt, alpha, beta, F_L96) = model_parameters 
    N_tilde = torch.zeros(N,N)
    L_tilde = torch.zeros(N,N)
    
    X = torch.zeros(N, ensemble_size)
    Chi = torch.zeros(N, ensemble_size)

    for i in range(ensemble_size):
        x_pert = torch.randn(N,1)*sd
        # print(x_pert)
        x_in_pert = x_out + x_pert
        x_out_pert = model(x_in_pert,model_parameters)
        x_out_unpert = model(x_out, model_parameters)
        Chi_i = x_out_pert - x_out_unpert
        # print("x_out_pert shape:", x_out_pert.shape)
        # print("x_out_unpert shape:", x_out_unpert.shape)
        # print("Chi_i shape:", Chi_i.shape)
        X[:,i] = x_pert.squeeze()
        Chi[:,i] = Chi_i.squeeze()

    for grid_index in range(N):
        X_i = member_past_selector(grid_index,X,N,ensemble_size)
        Chi_i = member_past_selector(grid_index,Chi,N,ensemble_size)

        n_i, l_i = little_IETLM(Chi_i, X_i)

        # print(n_i.shape)
        # print(l_i.shape)


        # print(M_i.shape)
        # print(sum(abs(Chi_i - M_i @ X_i)))
        # print(sum(abs(Chi_i[:,0] - M_i @ X_i[:,0])))
        # x_pert_test = torch.randn(N,1)*sd
        # x_pert_test_t1 = model(x_pert_test,model_parameters)


        # print(sum(abs(Chi_i[:,0] - M_i @ X_i[:,0])))
        # print(sum(abs(x_pert_test_t1 - M_i @ x_pert_test)))

        N_tilde = put_in_place(n_i, N, N_tilde, grid_index)
        L_tilde = put_in_place(l_i, N, L_tilde, grid_index)


    
    return  N_tilde, L_tilde

        
        



# def little_IETLM(Xi_i, X_i, ensemble_size):
#     # we first define the Pi half matrix
#     Pi_half = torch.cat((Xi_i, -X_i), dim = 1)
#     Pi = Pi_half @ Pi_half.T
#     eigenvals, eigenvecs = torch.lobpcg(Pi, 1)
#     n_i = eigenvecs[0,9]
#     l_i = eigenvecs[9,:]
#     return n_i, l_i

def little_IETLM(Xi_i, X_i):
    # Construct matrix
    # (1)
    Pi_half = torch.cat((Xi_i, -X_i), dim=0)
    Pi = Pi_half @ Pi_half.T
    # print(Pi_half.shape)

    # (2)
    # Compute smallest eigenpair
    eigenvals, eigenvecs = torch.lobpcg(Pi, k=1, largest=False)
    torch.linalg.eigvals(A)
    # print(eigenvals)

    # Smallest eigenvector (shape: [18])
    v = eigenvecs[:, 0]

    # (3)
    # Split into two 9-vectors
    n_i = v[:5]
    l_i = v[5:]

    return n_i, l_i

