# We start by importing the necessary packages as well as the python files that contain the time-stepping schemes.
import numpy as np
import time
import os
import argparse
import matplotlib.pyplot as plt
from mpi4py import MPI
import matplotlib.ticker as ticker
import math
import random
import cv2
import os






# We will now do step 1: Generate the ensemble data.
def generate_ensemble(M_1, M_2, N, ensemble_size, dt, dx, c, sd):
# We start by generating our initial vector
    Cov = cf.Cov_generator(N)
    u_g_1 = cf.inital_vector(N, Cov, 606).reshape(-1)   
    
    
    initial = np.zeros((N, ensemble_size))
    for i in range(ensemble_size):
        # We generate our smooth random noise
        delta_u_i = cf.inital_vector(N, Cov, 1475 + 3*(i**2)).reshape(-1) *10**(-6)
        # plt.plot(x_values, delta_u_i, marker='o', color='b', label='Random Vector')
        initial[:,i] = u_g_1 + delta_u_i
    
    # Now that we have our inital ensemble we wil run the model for one time step
    X = np.zeros((N, ensemble_size))
    for i in range(ensemble_size):
        psi =  initial[:,i]
        phi =  M_1(psi,4, N, dx, dt, c).reshape(-1) 
        X[:,i] = phi
    
    # Now that we have our inital ensemble we wil run the model for one time step
    Xi = np.zeros((N, ensemble_size))
    for i in range(ensemble_size):
        rho = initial[:,i]
        psi =  X[:,i]
        phi =  M_2(psi,rho, N, dx, dt, c).reshape(-1) 
        Xi[:,i] = phi
    
    # we will now do the same again
    Omega = np.zeros((N, ensemble_size))
    for i in range(ensemble_size):
        rho = X[:,i]
        psi = Xi[:,i]
        phi = M_2(psi, rho, N, dx, dt, c).reshape(-1) 
        Omega[:,i] = phi
    
    
    # We now need to remove the guess vector from the ensemble members
    u_g_2 = M_1(u_g_1,4, N, dx, dt, c).reshape(-1) 
    u_g_3 = M_2(u_g_2, u_g_1, N, dx, dt, c).reshape(-1) 
    u_g_4 = M_2(u_g_3, u_g_2, N, dx, dt, c).reshape(-1) 
    
    X_base = np.repeat(u_g_2[:, np.newaxis], ensemble_size, axis=1)
    Xi_base = np.repeat(u_g_3[:, np.newaxis], ensemble_size, axis=1)
    Omega_base = np.repeat(u_g_4[:, np.newaxis], ensemble_size, axis=1)
    
    Omega_prime = Omega - Omega_base
    X_prime = X - X_base
    Xi_prime = Xi - Xi_base
    return X_prime, Xi_prime, Omega_prime

    
    
    # We now need to remove the guess vector from the ensemble members
    u_g_2 = M_1(u_g_1,4, N, dx, dt, c).reshape(-1) 
    u_g_3 = M_2(u_g_2, u_g_1, N, dx, dt, c).reshape(-1) 
    u_g_4 = M_2(u_g_3, u_g_2, N, dx, dt, c).reshape(-1) 
    
    X_base = np.repeat(u_g_2[:, np.newaxis], ensemble_size, axis=1)
    Xi_base = np.repeat(u_g_3[:, np.newaxis], ensemble_size, axis=1)
    Omega_base = np.repeat(u_g_4[:, np.newaxis], ensemble_size, axis=1)
    
    Omega_prime = Omega - Omega_base
    X_prime = X - X_base
    Xi_prime = Xi - Xi_base
    return X_prime, Xi_prime, Omega_prime

# def generate_ensemble(M_1, M_2, N, ensemble_size, dt, dx, c, sd):
#     u = np.zeros((N,1))
#     Cov = cf.Cov_generator(N)
#     u = cf.inital_vector(N, Cov, 606).reshape(-1)   # (N,)
#     # We generate the ensemble that will be our time (i-1)
#     X =  cf.ensemble_generator(ensemble_size, N, sd, u)

#     u_dash =  M_1(u, u, N, dx, dt, c).reshape(-1)
#     u_dash_dash = M_2(u_dash, u, N, dx, dt, c).reshape(-1)
    
#     # We also generate the empty frames that hold our ensembles at time (i) and (i+1)
#     Xi = np.zeros((N, ensemble_size))
#     Omega = np.zeros((N, ensemble_size))
    
#     # We now run a step of the scheme for each ensemble member to get the ensemble at time (i)
#     for i in range(ensemble_size):
#         psi = X[:,i]
#         phi = M_1(psi, u, N, dx, dt, c).reshape(-1)
#         rho = M_2(phi, psi, N, dx, dt, c).reshape(-1)
#         # print(X[:,i].shape)
#         X[:,i] = psi - u
#         Xi[:,i] = phi - u_dash
#         Omega[:,i] = rho - u_dash_dash
#     return X, Xi, Omega

# def generate_ensemble(M_1, M_2, N, ensemble_size, dt, dx, c, sd):
#     u = np.zeros((N,1))
#     Cov = cf.Cov_generator(N)
#     u = cf.inital_vector(N,Cov,606).reshape((N,1)) *10**(-1)
#     # We generate the ensemble that will be our time (i-1)
#     X =  cf.ensemble_generator(ensemble_size, N, sd, u)
    
#     # We also generate the empty frames that hold our ensembles at time (i) and (i+1)
#     Xi = np.zeros((N, ensemble_size))
#     Omega = np.zeros((N, ensemble_size))
    
#     # We now run a step of the scheme for each ensemble member to get the ensemble at time (i)
#     for i in range(ensemble_size):
#         psi = X[:,i]
#         Xi[:,i] = M_1(psi, u, N, dx, dt, c).reshape(-1)
#         phi = Xi[:,i]
#         Omega[:,i] = M_2(phi, psi, N, dx, dt, c).reshape(-1)
#     return X, Xi, Omega

# We will now do step 2: remove the elements from the ensembles that relate to each variable j.

# We will define a function that takes the required elements from each time step

# This function grabs the (j-1) ,j, (j+1) and (j+2)
def member_grabber_past(j, X, ensemble_size,N):
    X_i = np.zeros((1,ensemble_size))
    X_i[0, :] = X[(j % N), :]
    return X_i

# This function grabs the j and the (j+1)
def member_grabber_current(j, Xi, ensemble_size,N):
    Xi_i = np.zeros((3,ensemble_size))
    Xi_i[0, :] = Xi[(j-1)%N,:]
    Xi_i[1, :] = Xi[(j) % N, :]
    Xi_i[2, :] = Xi[((j+1) % N), :]
    
    return Xi_i

def member_grabber_future(j, Omega, ensemble_size,N):
    Omega_i = np.zeros((3,ensemble_size))
    Omega_i[0, :] = Omega[(j-1)%N,:]
    Omega_i[1, :] = Omega[((j)) % N, :]
    Omega_i[2, :] = Omega[((j+1) % N), :]
    
    return Omega_i


# We will now do step three: compute \Pi_j

def Pi_half_generator(X_i, Xi_i, Omega_i, ensemble_size):
    Pi_half = np.vstack((Omega_i, -Xi_i, - X_i))/np.sqrt(ensemble_size)
    return Pi_half



def eigenvalue_calculator(Pi_half):
    Pi_full = Pi_half @ Pi_half.T
    eigenvalues, eigenvectors = np.linalg.eigh(Pi_full)    
    return eigenvalues, eigenvectors


# We will now do step 5: calculate which eigenvectors have zero-eigenvalue
def is_zero_eigenvector(eigenvalues, eigenvectors, threshold):
    total_eigenvalue = sum(abs(eigenvalues))
    zero_eigenvectors = []
    zero_eigenvalues = []
    
    for i in range(len(eigenvalues)):
        if abs(eigenvalues[i]) < threshold:
            zero_eigenvectors.append(eigenvectors[:, i])
            zero_eigenvalues.append(eigenvalues[i])

    return zero_eigenvectors, zero_eigenvalues
def three_smallest_eigenpairs(eigenvalues, eigenvectors, k=3):
    """
    Selects the k smallest eigenvalues by magnitude and their corresponding eigenvectors.
    
    Args:
        eigenvalues (array): 1D array of eigenvalues.
        eigenvectors (array): 2D array of eigenvectors (columns are eigenvectors).
        k (int): Number of smallest eigenvalues to select. Default is 3.
        
    Returns:
        selected_vectors (list of np.ndarray): The eigenvectors for the k smallest eigenvalues.
        selected_values (list of float): The k smallest eigenvalues.
    """
    # Sort indices by absolute eigenvalue
    idx = np.argsort(np.abs(eigenvalues))[:k]
    
    selected_vectors = [eigenvectors[:, i] for i in idx]
    selected_values  = [eigenvalues[i] for i in idx]
    
    return selected_vectors, selected_values

# We will now do step 6: generate N~, L~ and K~
def eigenvalue_splitter(zero_eigenvalue):
    n_tilde = zero_eigenvalue[0:3]
    l_tilde = zero_eigenvalue[3:6]
    k_tilde = zero_eigenvalue[6:7]
    return  n_tilde, l_tilde, k_tilde


# For the second part of step 6 we need to put n_tilde, l_tilde and k_tilde intot their respective places.
def put_in_place_future(j,n_tilde,N):
    temp = np.zeros((1,N))

    temp[0, (j-1)%N] = n_tilde[0]
    temp[0, (j)%N] = n_tilde[1]
    temp[0, (j+1)%N] = n_tilde[2]
    return temp

def put_in_place_current(j,l_tilde,N):
    temp = np.zeros((1,N))
    temp[0, (j-1)%N] = l_tilde[0]
    temp[0, j%N] = l_tilde[1]
    temp[0, (j+1)%N] = l_tilde[2]
    return temp

def put_in_place_past(j,k_tilde,N):
    temp = np.zeros((1,N))
    temp[0, j%N] = k_tilde[0]
    return temp
# To generate the N_tilde, L_tilde and K_tilde, we simply stack the outputs of this function.


# Step 7: We now need to take the inverses of N_tilde, L_tilde and K_tilde
def SVD_inverse(A):
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    A_inv =  VT.T @ np.linalg.inv(np.diag(S)) @U.T
    return A_inv


# We now combine steps 1-6 into one complete function
def GIETLM_generator(M_1, M_2, N, ensemble_size, dt, dx,c, sd, threshold):
    # We will now do step 1,: generate the ensemble_data
    X, Xi, Omega = generate_ensemble(M_1, M_2, N, ensemble_size, dt, dx, c, sd)

    # cf.mat_plot(X,'coolwarm','Covariance Matrix Heatmap')
    # cf.mat_plot(Xi,'coolwarm','Covariance Matrix Heatmap')
    # cf.mat_plot(Omega,'coolwarm','Covariance Matrix Heatmap')
    


    
    
    # # We will now do step 1: Generate the ensemble data.
    # u = np.zeros((N,1))
    # # We generate the ensemble that will be our time (i-1)
    # X =  cf.ensemble_generator(ensemble_size, N, sd, u)
    
    
    # # We also generate the empty frames that hold our ensembles at time (i)
    # Xi = np.zeros((N, ensemble_size))
    # Omega = np.zeros((N,ensemble_size))
    
    # N_tilde = np.zeros((1,N))
    L_tilde = np.zeros((1,N))
    N_tilde = np.zeros((1,N))
    K_tilde = np.zeros((1,N))



    eigen_saver = []
    
    # # We now run a step of the scheme for each ensemble member to get the ensemble at time (i)
    # for i in range(ensemble_size):
    #     psi = X[:,i].flatten()
    #     Xi[:,i] = M(psi, N, dx, dt, c).reshape(-1)
    #     phi = Xi[:,i]
    #     Omega[:,i] = M(phi, N, dx, dt, c).reshape(-1)
    
    for i in range(N):
        # We start by grabbing the non-zero elements of X, Xi and Omega
        X_i = member_grabber_past(i, X, ensemble_size,N)
        Xi_i = member_grabber_current(i, Xi, ensemble_size, N)
        Omega_i = member_grabber_current(i, Omega, ensemble_size, N)

        # if i == 3:
        #     # Plot the covariance matrix as a heatmap using matplotlib
        #     plt.figure(figsize=(8, 6))
        #     plt.imshow(Omega_i, cmap='coolwarm' , interpolation='nearest')
        #     plt.colorbar()  # Add color bar to indicate the scale of values
        #     plt.xlabel('Ensemble Member')  # label for x-axis
        #     plt.ylabel('Variable Number')  # label for y-axis
        #     # Display the heatmap
        #     plt.title('Omega_i for model variable i (ensemble at the future time)')
        #     plt.savefig(f'/Users/isaacmiles-fine/Documents/Honours/graphics4thesis/RK4_example/Omega_i.png',dpi=300, bbox_inches="tight")
        #     plt.show()


        # cf.mat_plot(X_i,'coolwarm','X_i')
        # cf.mat_plot(Xi_i,'coolwarm','Xi_i')
        
        # We now generate Pi^half_i
        Pi_i_half = Pi_half_generator(X_i, Xi_i, Omega_i, ensemble_size)
        # if i == 0:
        #     # Plot the covariance matrix as a heatmap using matplotlib
        #     plt.figure(figsize=(8, 6))
        #     plt.imshow(Pi_i_half, cmap='coolwarm' , interpolation='nearest')
        #     plt.colorbar()  # Add color bar to indicate the scale of values
        #     # plt.xlabel('Ensemble Member')  # label for x-axis
        #     # plt.ylabel('Variable Number')  # label for y-axis
        #     # Display the heatmap
        #     plt.title('Pi_i^(1/2)')
        #     plt.savefig(f'/Users/isaacmiles-fine/Documents/Honours/graphics4thesis/RK4_example/Pi_half.png',dpi=300, bbox_inches="tight")
        #     plt.show()

        # print(Pi_i_half)
        # We now compute the eigenvalues of Pi^half_i
        eigenvalues, eigenvectors = eigenvalue_calculator(Pi_i_half)
        # print("eigenvalues at", i,eigenvalues)
        eigen_saver.append(abs(eigenvalues))


        
        # We now find the eigenvectors that have zero_eigenvale
        if M_2 == TST.Runge_Kutta_4_Spectral_L:
            zero_eigenvectors, zero_eigenvalues = three_smallest_eigenpairs(eigenvalues, eigenvectors, k=2)
            # print('Hello Good Sir')
        elif M_2 == TST.Runge_Kutta_4_Spectral_NL:
            zero_eigenvectors, zero_eigenvalues = three_smallest_eigenpairs(eigenvalues, eigenvectors, k=4)
            # print('Hello Good Sir')
        elif M_2 == CN.CN_step:
            zero_eigenvectors, zero_eigenvalues = three_smallest_eigenpairs(eigenvalues, eigenvectors, k=1)
            # print('Hello Good Sir')
                  
        else:
            zero_eigenvectors, zero_eigenvalues = is_zero_eigenvector(eigenvalues, eigenvectors, threshold)
        # print("zero_eigenvectors:",zero_eigenvalues)
        # print(len(zero_eigenvectors))
        for eig in range(len(zero_eigenvectors)):
                
            # print(i, eig)
            zero_eigenvector = zero_eigenvectors[eig]
            # zero_eigenvector = [1,2,3,4,5,6]
            # print(zero_eigenvector)
            
            # Test function
            # zero_eigenvector = np.arange(6)+1

            
            n_tilde_p, l_tilde_p, k_tilde_p = eigenvalue_splitter(zero_eigenvector)

            # print(zero_eigenvector - np.hstack((n_tilde_p, l_tilde_p)))
            # Test case:
            X_i_test = member_grabber_past(i, X, ensemble_size,N)[:,3]
            Xi_i_test = member_grabber_current(i, Xi, ensemble_size,N)[:,3]
            Omega_i_test = member_grabber_future(i, Omega, ensemble_size,N)[:,3]

            test = n_tilde_p @ Omega_i_test - l_tilde_p @ Xi_i_test - k_tilde_p @ X_i_test
            # print("Test result:",test)

            
            # print("n_tilde_p", n_tilde_p)
            # print("l_tilde_p", l_tilde_p)
            n_tilde = put_in_place_future(i,n_tilde_p,N)
            l_tilde = put_in_place_current(i,l_tilde_p,N)
            k_tilde = put_in_place_past(i,k_tilde_p, N)
            
            # print(n_tilde.shape)
            # print(n_tilde)
            N_tilde = np.vstack((N_tilde, n_tilde))
            L_tilde = np.vstack((L_tilde, l_tilde))
            K_tilde = np.vstack((K_tilde, k_tilde))


    N_tilde = N_tilde[1:, :]
    L_tilde = L_tilde[1:, :]
    K_tilde = K_tilde[1:, :]

    
    
            
    return N_tilde, L_tilde, K_tilde, eigen_saver
