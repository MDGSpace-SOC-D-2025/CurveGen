import CurveGen_dataset
'''print("DATASET FILE:", CurveGen_dataset.__file__)
print("HAS test_data:", hasattr(CurveGen_dataset, "test_data"))
print("DIR:", dir(CurveGen_dataset))'''
import numpy as np
from CurveGen_dataset import test_data

x_data, y_data = test_data(noise=0) 
def build_baseTerms(x):
    x_clean=np.abs(x)+1e-9
    func_dict = {
        "1":      np.ones_like(x),
        "x":      x,
        "x^2":    x**2,
        "1/x":    1.0/x_clean,
        "sin":    np.sin(x),
        "cos":    np.cos(x),
        "exp":    np.exp(x),
        "log":    np.log(x_clean),
        "1/exp":  np.exp(x*(-1)),
        "1/exp2": np.exp(x*x*(-1)),
        "sqrt":   np.sqrt(x_clean)
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
            elif k1=="x" and k2=="x": continue
            elif k1=="x^2" and k2=="1/x": continue
            else: name, val = f"{k1}*{k2}", func_dict[k1] * func_dict[k2]
            
            if name not in term_name:
                term_name.append(name)
                term_valv.append(val)
    
    return term_name, term_valv

def reduction_function(x,y):
    term_name,term_valv=build_baseTerms(x)
    term_name=np.array(term_name)
    dominant_mask=[]
    possible_dominants=[]
    possible_dominants_r_value=[]
    r_values=np.zeros(len(term_name))

    for i in range(len(term_name)):
        r_values[i]=np.corrcoef(y,term_valv[i])[0,1]
    dominant_mask=r_values>0.95
    possible_dominants=term_name[dominant_mask]
    possible_dominants_r_value=r_values[dominant_mask]
    print(possible_dominants)
    print(possible_dominants_r_value)

        
   

reduction_function(x_data,y_data)
