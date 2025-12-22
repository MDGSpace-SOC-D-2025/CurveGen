import numpy as np

def test_data(noise=0.0):
    test_data_min = 1
    test_data_max = 10
    test_values = 100
    x_test_data = np.linspace(test_data_min, test_data_max, test_values)    
    y_test_data = x_test_data**2+np.sin(x_test_data)
    
    if noise > 0:
        y_test_data += np.random.normal(0, noise, size=len(y_test_data))

    return x_test_data, y_test_data