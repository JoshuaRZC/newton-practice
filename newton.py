import math

def newton(x0, fun, eps=1e-5, tol = 1e-5):
    x_new = x0
    x_old = math.inf
    while abs(x_new - x_old) > tol:
        first_derivative = (fun(x_new + eps) - fun(x_new)) / eps
        second_derivative = ((fun(x_new + 2 * eps) - fun(x_new + eps)) / eps - (fun(x_new + eps) - fun(x_new)) / eps) / eps
        x_old = x_new
        x_new = x_old - first_derivative / second_derivative
    return x_new