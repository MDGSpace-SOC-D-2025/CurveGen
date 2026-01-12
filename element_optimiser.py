import numpy as np
from scipy.optimize import curve_fit
from CurveGen_dataset import test_data

x_data, y_data = test_data(noise=1)

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

lowest_error = float('inf') 
best_result_name = "None"

try:
    popt, pcov = curve_fit(linear, x_data, y_data)
    error = np.sum((y_data - linear(x_data, *popt))**2)
    
    if error < lowest_error:
        lowest_error = error
        best_result_name = "linear"
except RuntimeError:
    pass 

try:
    popt, pcov = curve_fit(quadratic, x_data, y_data)
    error = np.sum((y_data - quadratic(x_data, *popt))**2)
    
    if error < lowest_error:
        lowest_error = error
        best_result_name = "quadratic"
except RuntimeError:
    pass

try:
    popt, pcov = curve_fit(power, x_data, y_data)
    error = np.sum((y_data - power(x_data, *popt))**2)
    
    if error < lowest_error:
        lowest_error = error
        best_result_name = "power"
except RuntimeError:
    pass

try:
    guess_offset = np.mean(y_data)
    guess_amp = (np.max(y_data) - np.min(y_data)) / 2
    
    fft_spectrum = np.fft.rfft(y_data - guess_offset)
    fft_freqs = np.fft.rfftfreq(len(y_data), d=(x_data[1]-x_data[0]))
    
    peak_idx = np.argmax(np.abs(fft_spectrum))
    guess_freq = fft_freqs[peak_idx]
    guess_k = 2 * np.pi * guess_freq
    if guess_k == 0: guess_k = 1.0

    p0_sine = [guess_amp, guess_k, 0, guess_offset]

    popt, pcov = curve_fit(sine, x_data, y_data, p0=p0_sine, maxfev=5000)
    error = np.sum((y_data - sine(x_data, *popt))**2)
    
    if error < lowest_error:
        lowest_error = error
        best_result_name = "sine"
except RuntimeError:
    pass

try:
    popt, pcov = curve_fit(exponential, x_data, y_data)
    error = np.sum((y_data - exponential(x_data, *popt))**2)
    
    if error < lowest_error:
        lowest_error = error
        best_result_name = "exponential"
except RuntimeError:
    pass

try:
    popt, pcov = curve_fit(inverse, x_data, y_data)
    error = np.sum((y_data - inverse(x_data, *popt))**2)
    
    if error < lowest_error:
        lowest_error = error
        best_result_name = "inverse"
except RuntimeError:
    pass

try:
    popt, pcov = curve_fit(inverse_square, x_data, y_data)
    error = np.sum((y_data - inverse_square(x_data, *popt))**2)
    
    if error < lowest_error:
        lowest_error = error
        best_result_name = "inverse_square"
except RuntimeError:
    pass

try:
    popt, pcov = curve_fit(sqrt, x_data, y_data)
    error = np.sum((y_data - sqrt(x_data, *popt))**2)
    
    if error < lowest_error:
        lowest_error = error
        best_result_name = "sqrt"
except RuntimeError:
    pass

try:
    popt, pcov = curve_fit(inverseSQRT, x_data, y_data)
    error = np.sum((y_data - inverseSQRT(x_data, *popt))**2)
    
    if error < lowest_error:
        lowest_error = error
        best_result_name = "inverseSQRT"
except RuntimeError:
    pass

print(f"Best Fit Type: {best_result_name}")
print(f"Error (SSR): {lowest_error}")