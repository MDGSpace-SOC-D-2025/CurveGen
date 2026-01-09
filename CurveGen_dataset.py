import numpy as np

unit_list=['kg','m','s','A','K']
unit_dimensions={(1.0,):'Mass',(0.0,1.0):'Length',(0.0, 0.0, 1.0): 'Time',
    (0.0, 0.0, 0.0, 1.0): 'Current',
    (0.0, 0.0, 0.0, 0.0, 1.0): 'Temperature',
    (0.0, 2.0): 'Area',
    (0.0, 3.0): 'Volume',
    (0.0, 1.0, -1.0): 'Velocity',
    (0.0, 1.0, -2.0): 'Acceleration',
    (1.0, 1.0, -2.0): 'Force',
    (1.0, 2.0, -2.0): 'Energy',
    (1.0, 2.0, -3.0): 'Power',
    (1.0, -1.0, -2.0): 'Pressure',
    (1.0, -3.0): 'Density',
    (1.0, 1.0, -1.0): 'Momentum',
    (0.0, 0.0, -1.0): 'Frequency',
    (0.0, 0.0, 1.0, 1.0): 'Charge',
    (1.0, 2.0, -3.0, -1.0): 'Voltage',
    (1.0, 2.0, -3.0, -2.0): 'Resistance',
    (-1.0, -2.0, 4.0, 2.0): 'Capacitance',
    (0.0, 0.0, -1.0): 'Angular Velocity',
    (0.0, 0.0, -2.0): 'Angular Acceleration',
    (1.0, 2.0): 'Moment of Inertia',
    (1.0, 2.0, -2.0): 'Torque',
    (1.0, 0.0, -2.0): 'Surface Tension',
    (1.0, -1.0, -1.0): 'Viscosity',
    (0.0, 2.0, -2.0, 0.0, -1.0): 'Specific Heat',
    (1.0, 1.0, -3.0, 0.0, -1.0): 'Thermal Conductivity',
    (1.0, 2.0, -2.0, 0.0, -1.0): 'Entropy',
    (1.0, 1.0, -3.0, -1.0): 'Electric Field',
    (1.0, 2.0, -2.0, -1.0): 'Magnetic Flux',
    (1.0, 0.0, -2.0, -1.0): 'Magnetic Flux Density',
    (1.0, 2.0, -2.0, -2.0): 'Inductance',
    (1.0, 2.0, -1.0): 'Planck Constant',
    (-1.0, 3.0, -2.0): 'Gravitational Constant'}
def get_dimensions(a="text"):
        dimensions = np.array(input(a).split(), dtype=float)
        return dimensions
def test_data(noise=0.0):
    
        
    
    test_data_min = 1
    test_data_max = 10
    test_values = 100
    x_test_data = np.linspace(test_data_min, test_data_max, test_values)    
    y_test_data = 4*np.sqrt(3*x_test_data+2)+10
    '''
    x_test_data = np.array(input("Enter the values of one of the parameters: ").split(), dtype=float)
    x_dimensions=get_dimensions("input the unit for the parameter in the form of M L T I K")
    x_dimensions_tuple=tuple(x_dimensions.tolist())
    parts = [f"{t}[{c:.3f}]" for c, t in zip(x_dimensions, unit_list)]
    print(f'Physical quantity is {unit_dimensions[x_dimensions_tuple]},'+' unit:'+"".join(parts))

    y_test_data = np.array(input("Enter result values: ").split(), dtype=float)
    print(y_test_data)
    y_dimensions=get_dimensions("input the unit for the results in the form of M L T I K")
    y_dimensions_tuple=tuple(y_dimensions.tolist())
    parts = [f"{t}[{c:.3f}]" for c, t in zip(y_dimensions, unit_list)]
    print(f'Physical quantity is {unit_dimensions[y_dimensions_tuple]},'+' unit:'+"".join(parts))'''
    if noise > 0:
        y_test_data += np.random.normal(0, noise, size=len(y_test_data))

    return x_test_data, y_test_data

def InputData():
    paramters_no=int(input('enter the number of parameters you have data for'))
    data={}
    parameter_dimensions=[]

    for i in range(paramters_no):
        dimensions=get_dimensions("input the unit for the parameter in the form of M L T I K: ")
        dimensions_tuple=tuple(dimensions.tolist())
        parts = [f"{t}[{c:.3f}]" for c, t in zip(dimensions, unit_list)]
        print(f'Physical quantity is {unit_dimensions[dimensions_tuple]},'+' unit:'+"".join(parts))
        parameter_dimensions.append(dimensions_tuple)
        parameter_data= np.array(input(f"Enter the data for {unit_dimensions[dimensions_tuple]} with a gap between each data point: ").split(), dtype=float)
        data[unit_dimensions[dimensions_tuple]]=parameter_data
    print(data) 
    parameter_dimensions_set=set(parameter_dimensions)
    print(parameter_dimensions_set) 
    
    return data,parameter_dimensions_set



def build_baseTerms(x):
    x_clean=np.abs(x)+1e-9
    func_dict = {
        "1":      np.ones_like(x),
        "x":      x,
        "x^2":    x**2,
        "1/x":    1.0/x_clean,
        "sqrt":   np.sqrt(x),
        "sin":    np.sin(x),
        "cos":    np.cos(x),
        "exp":    np.exp(x),
        "log":    np.log(x_clean),
        "1/exp":  np.exp(x*(-1)),
        "1/exp2": np.exp(x*x*(-1))
    }
    
    term_name = []
    term_valv = []
    keys = list(func_dict.keys())

    for i in range(len(keys)):
        for j in range(i, len(keys)):
            k1 = keys[i]
            k2 = keys[j]
            
            if k1 == "1" and k2 == "1": name,val= "1",func_dict['1']
            if k1 == "1":    name, val = k2, func_dict[k2]
            elif k2 == "1":  name, val = k1, func_dict[k1]
            elif k1=="x" and k2=="1/x": continue
            elif k1=="x" and k2=="1/x": continue
            elif k1=="exp" and k2=="1/exp": continue
            elif k1=="1/exp" and k2=="1/exp": continue
            elif k1=="1/exp" and k2=="1/exp2": continue
            elif k1=="1/exp2" and k2=="1/exp2": continue
            elif k1=="exp" and k2=="1/exp2": continue
            elif k1=="sqrt" and k2=="sqrt": continue
            elif k1=="1/x" and k2=="1/x": continue
            elif k1=="cos" and k2=="log": continue
            elif k1=="sin" and k2=="log": continue
            elif k1=="sqrt" and k2=="log": continue
            elif k1=="sqrt" and k2=="exp": continue
            elif k1=="sqrt" and k2=="sin": continue
            elif k1=="sqrt" and k2=="cos": continue
            elif k1=="log" and k2=="log": continue
            elif k1=="x" and k2=="x": continue
            elif k1=="x^2" and k2=="1/x": continue
            else: name, val = f"{k1}*{k2}", func_dict[k1] * func_dict[k2]
            
            if name not in term_name:
                term_name.append(name)
                term_valv.append(val)

    return term_name, term_valv

