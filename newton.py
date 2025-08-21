import math

def get_first_derivative(x, fun, eps=1e-5):
    """
    Calculate the first derivative of a function at a point x.

    Parameters
    ---------------------
    x : float
        Point to compute the second derivative
    fun : callable
        Function to compute the second derivative
    eps : float, optional
        Small value for numerical differentiation (default is 1e-5).

    Returns:
    ---------------------
    float
        First derivative of the function at point x.
    """
    return (fun(x + eps) - fun(x)) / eps

def get_second_derivative(x, fun, eps=1e-5):
    """
    Calculate the second derivative of a function at a point x.

    Parameters
    ---------------------
    x : float
        Point to compute the second derivative
    fun : callable
        Function to compute the second derivative
    eps : float, optional
        Small value for numerical differentiation (default is 1e-5).

    Returns
    ---------------------
    float
        Second derivative of the function at point x.
    """
    return (get_first_derivative(x + eps, fun, eps) - get_first_derivative(x, fun, eps)) / eps

def minimize(x0, fun, max_iter=100, eps=1e-5, tol=1e-5, alpha=0.25, beta=0.5):
    """
    Minimize a convex function using Newton's method with backtracking line search.

    Parameters:
    ---------------------
    x0 : float
        Initial guess for the minimum.
    fun : callable
        Function to minimize.
    max_iter : int
        Maximum number of iterations.
    eps : float, optional
        Small value for numerical differentiation (default is 1e-5).
    tol : float, optional
        Tolerance for convergence (default is 1e-5).
    alpha : float, optional
        Parameter for the Armijo condition (default is 0.25).
    beta : float, optional
        Parameter for the backtracking line search (default is 0.5).

    Returns:
    ---------------------
    x_new : float
        Estimated location of the minimum.
    """
    if not callable(fun):
        raise ValueError("The function to minimize must be callable.")
    if not isinstance(x0, (int, float)):
        raise ValueError("Initial guess x0 must be a number.")
    if max_iter <= 0:
        raise ValueError("Maximum number of iterations must be a positive integer.")
    if eps <= 0:
        raise ValueError("Epsilon must be a positive number.")
    if tol <= 0:
        raise ValueError("Tolerance must be a positive number.")
    x_new = x0
    x_old = math.inf
    n_iter = 0
    while abs(x_new - x_old) > tol:
        first_derivative = get_first_derivative(x_new, fun, eps)
        second_derivative = get_second_derivative(x_new, fun, eps)
        if second_derivative == 0:
            print("Warning: Second derivative is zero, cannot proceed with Newton's method.")
            break
        elif second_derivative < 0:
            print("Warning: Function is not convex, cannot proceed with Newton's method.")
            break
        search_direction = - first_derivative / second_derivative
        x_old = x_new

        # Backtracking Line Search
        t = 1
        x_new = x_old + t * search_direction
        while fun(x_new) > fun(x_old) + alpha * t * first_derivative * search_direction:
            t = beta * t
            x_new = x_old + t * search_direction
        n_iter += 1

        if n_iter == max_iter:
            print("Warning: Maximum number of iterations reached.")
            break
    return x_new