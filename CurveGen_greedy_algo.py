import CurveGen_dataset
import numpy as np
from CurveGen_dataset import test_data

x_data, y_data = test_data(noise=0.5) 
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
            elif k1=="x" and k2=="x": continue
            elif k1=="x^2" and k2=="1/x": continue
            else: name, val = f"{k1}*{k2}", func_dict[k1] * func_dict[k2]
            
            if name not in term_name:
                term_name.append(name)
                term_valv.append(val)
    
    return term_name, term_valv

def greedy_algo(x, y):
    
    term_name,term_valv=build_baseTerms(x)
    current_error = np.var(y) 
    
    found_indices = []
    found_cols = []
    
    for step in range(5):
        best_idx = -1
        best_improvement = 0.0        
        for i in range(len(term_name)):
            if i in found_indices: continue 
            current_cols = found_cols + [term_valv[i]]
            A_test = np.column_stack(current_cols)

            coeffs, resid, _, _ = np.linalg.lstsq(A_test, y, rcond=None)
            
            if resid.size > 0:
                y_pred = A_test @ coeffs
                new_error = np.var(y - y_pred)
            else:
                new_error = 0.0
            improvement = (current_error - new_error) / current_error
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_idx = i
        
        if best_improvement < 0.05: 
            break
            
        found_indices.append(best_idx)
        found_cols.append(term_valv[best_idx])
        A_accepted = np.column_stack(found_cols)
        coeffs, _, _, _ = np.linalg.lstsq(A_accepted, y, rcond=None)
        y_pred = A_accepted @ coeffs
        current_error = np.var(y - y_pred)
        if current_error < 1e-9: break
    if not found_indices: return [], []
    
    A_final = np.column_stack(found_cols)
    final_coeffs, _, _, _ = np.linalg.lstsq(A_final, y, rcond=None)
    
    return final_coeffs, [term_name[i] for i in found_indices]


coeffs, terms = greedy_algo(x_data, y_data)

parts = [f"{c:.3f}[{t}]" for c, t in zip(coeffs, terms)]
print(" + ".join(parts))

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
