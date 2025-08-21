import numpy as np
import jax

def get_gradient(x, fun):
    """
    Calculate the gradient of a multivariate function at a point x.

    Parameters
    ---------------------
    x : list or np.ndarray
        Point to compute the gradient
    fun : callable
        Function to compute the gradient

    Returns:
    ---------------------
    list
        Gradient of the function at point x.
    """
    x_jax = jax.numpy.array(x)
    gradient = jax.grad(fun)(x_jax)
    return np.array(gradient)

def get_hessian(x, fun):
    """
    Calculate the hessian of a multivariate function at a point x.

    Parameters
    ---------------------
    x : list or np.ndarray
        Point to compute the hessian
    fun : callable
        Function to compute the hessian

    Returns:
    ---------------------
    list
        Hessian of the function at point x.
    """
    x_jax = jax.numpy.array(x)
    hessian = jax.hessian(fun)(x_jax)
    return np.array(hessian)

def multivariate_newton(x0, fun, max_iter=100, tol=1e-5):
    """
    Minimize a multivariate convex function using Newton's method.

    Parameters:
    ---------------------
    x0 : list or np.ndarray
        Initial guess for the optimize.
    fun : callable
        Function to optimize.
    max_iter : int
        Maximum number of iterations.
    tol : float, optional
        Tolerance for convergence (default is 1e-5).

    Returns:
    ---------------------
    x_new : np.ndarray
        Estimated location of the minimum.
    """
    
    if not callable(fun):
        raise ValueError("The function to minimize must be callable.")
    if not isinstance(x0, (list, np.ndarray)):
        raise ValueError("Initial guess x0 must be a list or numpy array.")
    if max_iter <= 0:
        raise ValueError("Maximum number of iterations must be a positive integer.")
    if tol <= 0:
        raise ValueError("Tolerance must be a positive number.")
    
    x_new = np.array(x0, dtype=float)
    x_old = np.inf
    n_iter = 0

    while np.linalg.norm(x_new - x_old) > tol:
        gradient = get_gradient(x_new, fun)
        hessian_matrix = get_hessian(x_new, fun)

        if np.linalg.det(hessian_matrix) < 1e-10:
            print("Warning: Singular Hessian, cannot proceed with Newton's method.")
            break

        search_direction = - np.linalg.solve(hessian_matrix, gradient)
        x_old = x_new
        x_new = x_old + search_direction
        n_iter += 1
        
        if n_iter == max_iter:
            print("Warning: Maximum number of iterations reached.")
            break
            
    return x_new