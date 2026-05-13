
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import torch
from tqdm import tqdm
import math

# def lorenz96(u, F_L96):
#     """Standard Lorenz-96 tendency."""
#     return (torch.roll(u, -1) - torch.roll(u, 2)) * torch.roll(u, 1) - u + F_L96

# def source_s(u):
#     """Source term s(u) — define as needed."""
#     return 0.1 * u  # Example: linear in u

def source_s(u):
    u = torch.asarray(u)

    M = torch.zeros((len(u), len(u)))

    # find closest indices to 10 and 30
    i1 = 10
    i2 = 30

    M[i1, i1] = 1
    M[i2, i2] = 1

    # print(M)
    
    return torch.matmul(M, ( u**2))
    
    

def advection_upwind(C, u, dx):
    """Upwind advection: -u * dC/dx with periodic BCs."""
    dCdx = torch.where(
        u >= 0,
        (C - torch.roll(C, 1)) / dx,   # Backward diff for u > 0
        (torch.roll(C, -1) - C) / dx   # Forward diff for u < 0
    )
    return -u * dCdx


def rhs(y, model_parameters):
    (N, dx, dt, alpha, beta, F_L96) = model_parameters 
    C = y[:N]
    u = y[N:]
    
    # 1. Advection and Source remain (C depends on u)
    dCdt = advection_upwind(C, u, dx) + source_s(u) - alpha * C
    
    # 2. Modified Lorenz-96 (u depends strongly on C)
    # Standard L96 terms
    L96 = (torch.roll(u, -1) - torch.roll(u, 2)) * torch.roll(u, 1) - u + F_L96
    
    # Strong coupling: Let C modulate the forcing or act as a non-linear drag
    # Example: beta * C * u creates a "state-dependent" damping
    drag = beta * C * u 
    
    dudt = L96 - drag
    
    return torch.cat([dCdt, dudt])


def rhs_L96(u, model_parameters):
    (N, dx, dt, alpha, beta, F_L96) = model_parameters 
    
    L96 = (torch.roll(u, -1) - torch.roll(u, 2)) * torch.roll(u, 1) - u + F_L96
    return L96  


def rk4_step(y, model_parameters):
    (N, dx, dt, alpha, beta, F_L96) = model_parameters 
    k1 = rhs(y, model_parameters)
    k2 = rhs(y + 0.5 * dt * k1, model_parameters)
    k3 = rhs(y + 0.5 * dt * k2, model_parameters)
    k4 = rhs(y + dt * k3, model_parameters)
    
    return y + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

def rk4_L96(y, model_parameters):
    (N, dx, dt, alpha, beta, F_L96) = model_parameters 
    k1 = rhs_L96(y, model_parameters)
    k2 = rhs_L96(y + 0.5 * dt * k1, model_parameters)
    k3 = rhs_L96(y + 0.5 * dt * k2, model_parameters)
    k4 = rhs_L96(y + dt * k3, model_parameters)
    return y + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)


# Here is the code for the Crank-Nicolson Version of the Lorenz 96 model
def rhs_L96(u, model_parameters):
    (N, dx, dt, alpha, beta, F_L96) = model_parameters 
    return (torch.roll(u, -1) - torch.roll(u, 2)) * torch.roll(u, 1) - u + F_L96






# From Gemini

# def CN_L96(x_in, model_parameters, iter_count=5):
#     (N, dx, dt, alpha, beta, F_96) = model_parameters 
#     """
#     Performs a single Crank-Nicolson integration step for Lorenz 96.
    
#     Args:
#         x_n (torch.Tensor): Current state vector of shape (N,)
#         F (float): Forcing constant
#         dt (float): Time step size
#         iter_count (int): Number of fixed-point iterations for the implicit solve
        
#     Returns:
#         torch.Tensor: The state at x_{n+1}
#     """
    
#     def f(x):
#         # L96: (x[i+1] - x[i-2]) * x[i-1] - x[i] + F
#         return (torch.roll(x, -1) - torch.roll(x, 2)) * torch.roll(x, 1) - x + F_96

#     # Pre-calculate the explicit portion: x_n + (dt/2) * f(x_n)
#     explicit_part = x_in + (dt / 2.0) * f(x_in)
    
#     # Initialize x_next (using x_n as the initial guess)
#     x_next = x_in.clone()
    
#     # Implicit solve via fixed-point iteration
#     for _ in range(iter_count):
#         x_next = explicit_part + (dt / 2.0) * f(x_next)
        
#     return x_next


# def CN_L96(x_in, model_parameters, iter_count=5, tol=1e-6):
#     """
#     Performs a single Crank-Nicolson integration step for Lorenz 96 
#     using a Newton-Raphson solver.
#     """
#     (N, dx, dt, alpha, beta, F_96) = model_parameters 
    
#     def f(x):
#         # Lorenz 96 dynamics
#         return (torch.roll(x, -1) - torch.roll(x, 2)) * torch.roll(x, 1) - x + F_96

#     # Constant part of the CN equation
#     explicit_part = x_in + (dt / 2.0) * f(x_in)
    
#     # Initial guess for x_{n+1}
#     x_next = x_in.clone().requires_grad_(True)
    
#     for _ in range(iter_count):
#         # Define the residual G(x_next) = 0
#         residual = x_next - (dt / 2.0) * f(x_next) - explicit_part
        
#         # Check for convergence early to save computation
#         if torch.norm(residual) < tol:
#             break
            
#         # Compute the Jacobian J = dG/dx
#         # G is shape (N,), x_next is shape (N,)
#         jacobian = torch.autograd.functional.jacobian(
#             lambda x: x - (dt / 2.0) * f(x) - explicit_part, 
#             x_next
#         )
        
#         # Solve the linear system: J * delta = -residual
#         # delta = -inv(J) * residual
#         delta = torch.linalg.solve(jacobian, -residual)
        
#         # Update estimate
#         with torch.no_grad():
#             x_next += delta
            
#     return x_next.detach()
def CN_L96(x_in, model_parameters, iter_count=5, tol=1e-6):
    (N, dx, dt, alpha, beta, F_96) = model_parameters 
    
    def f(x):
        # Lorenz 96 dynamics
        return (torch.roll(x, -1) - torch.roll(x, 2)) * torch.roll(x, 1) - x + F_96

    # 1. Ensure x_in is 1D (shape: [N])
    x_in = x_in.flatten()
    
    explicit_part = x_in + (dt / 2.0) * f(x_in)
    x_next = x_in.clone().requires_grad_(True)
    
    for _ in range(iter_count):
        # f(x_next) and explicit_part are now both 1D
        residual = x_next - (dt / 2.0) * f(x_next) - explicit_part
        
        if torch.norm(residual) < tol:
            break
            
        # 2. Jacobian will now be (N, N) because x_next is (N,)
        jacobian = torch.autograd.functional.jacobian(
            lambda x: x - (dt / 2.0) * f(x) - explicit_part, 
            x_next
        )
        
        # 3. Solve (both jacobian and residual are now correctly shaped)
        # We use -residual.unsqueeze(-1) if we want a column back, 
        # but solve handles (N, N) and (N) perfectly.
        delta = torch.linalg.solve(jacobian, -residual)
        
        with torch.no_grad():
            x_next += delta
            
    # Return it in the original shape if necessary (e.g., [N, 1])
    return x_next.detach().view(-1, 1)
