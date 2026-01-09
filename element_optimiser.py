import numpy as np
from scipy.optimize import curve_fit
from CurveGen_dataset import test_data

x_data, y_data = test_data(noise=0) 

def linear(x, A, B):
    return A*x + B

def quadratic(x, A, B, C):
    return A*x*x + B*x + C

def power(x, A, n, C):        
    return A * np.power(x, n) + C

def sine(x, A, k, p, C):  
    return A * np.sin(k*x + p) + C

def exponential(x, A, k, C):  
    return A * np.exp(k*x) + C

def inverse(x, A, B, C):         
    return (A / (x + B)) + C

def inverse_square(x, A, C):        
    return A / (x**2) + C

def sqrt(x, a, b, c):
    return a * np.sqrt(x + b) + c

def inverseSQRT(x, a, b, c):
    return a / (np.sqrt(x + b)) + c

lowestError = float('inf') 
BestType = "None"
best_expression = "None"

try:
    popt, pcov = curve_fit(linear, x_data, y_data)
    a, b = popt
    error = np.sum((y_data - linear(x_data, *popt))**2)
    
    if error < lowestError:
        lowestError = error
        BestType = "linear"
        best_expression = f"{a:.3f}x + {b:.3f}"
except RuntimeError:
    pass 

try:
    popt, pcov = curve_fit(quadratic, x_data, y_data)
    a, b, c = popt
    error = np.sum((y_data - quadratic(x_data, *popt))**2)
    
    if error < lowestError:
        lowestError = error
        BestType = "quadratic"
        best_expression = f"{a:.3f}x^2 + {b:.3f}x + {c:.3f}"
except RuntimeError:
    pass

try:
    popt, pcov = curve_fit(power, x_data, y_data)
    a, n, c = popt
    error = np.sum((y_data - power(x_data, *popt))**2)
    
    if error < lowestError:
        lowestError = error
        BestType = "power"
        best_expression = f"{a:.3f}x^{n:.3f} + {c:.3f}"
except RuntimeError:
    pass

try:
    popt, pcov = curve_fit(sine, x_data, y_data, p0=[1, 1, 0, 0]) # p0 helps sine converge
    a, k, p, c = popt
    error = np.sum((y_data - sine(x_data, *popt))**2)

    
    if error < lowestError:
        lowestError = error
        BestType = "sine"
        best_expression = f"{a:.3f}sin({k:.3f}x + {p:.3f}) + {c:.3f}"
except RuntimeError:
    pass

try:
    popt, pcov = curve_fit(exponential, x_data, y_data)
    a, k, c = popt
    error = np.sum((y_data - exponential(x_data, *popt))**2)
    
    if error < lowestError:
        lowestError = error
        BestType = "exponential"
        best_expression = f"{a:.3f}e^({k:.3f}x) + {c:.3f}"
except RuntimeError:
    pass

try:
    popt, pcov = curve_fit(inverse, x_data, y_data)
    a, b, c = popt
    error = np.sum((y_data - inverse(x_data, *popt))**2)
    
    if error < lowestError:
        lowestError = error
        BestType = "inverse"
        best_expression = f"{a:.3f}/(x + {b:.3f}) + {c:.3f}"
except RuntimeError:
    pass

try:
    popt, pcov = curve_fit(inverse_square, x_data, y_data)
    a, c = popt
    error = np.sum((y_data - inverse_square(x_data, *popt))**2)
    
    if error < lowestError:
        lowestError = error
        BestType = "inverse_square"
        best_expression = f"{a:.3f}/x^2 + {c:.3f}"
except RuntimeError:
    pass

try:
    popt, pcov = curve_fit(sqrt, x_data, y_data)
    a, b, c = popt
    error = np.sum((y_data - sqrt(x_data, *popt))**2)
    
    if error < lowestError:
        lowestError = error
        BestType = "sqrt"
        best_expression = f"{a:.3f}sqrt(x + {b:.3f}) + {c:.3f}"
except RuntimeError:
    pass

try:
    popt, pcov = curve_fit(inverseSQRT, x_data, y_data)
    a, b, c = popt
    error = np.sum((y_data - inverseSQRT(x_data, *popt))**2)
    
    if error < lowestError:
        lowestError = error
        BestType = "inverseSQRT"
        best_expression = f"{a:.3f}/sqrt(x + {b:.3f}) + {c:.3f}"
except RuntimeError:
    pass

print(f"Best Fit Type: {BestType}")
print(f"Error: {lowestError}")
print(f"Expression: {best_expression}")





