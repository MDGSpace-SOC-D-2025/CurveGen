import numpy as np
from CurveGen_dataset import test_data,build_baseTerms

x_data, y_data = test_data(noise=0) 

def mse(residuals):
    m=np.mean(residuals**2)
    return m

def greedy_algo(x, y):
    
    term_name,term_valv=build_baseTerms(x)
    current_error = mse(y) 
    found_indices = []
    found_cols = []
    
    for step in range(3):
        best_idx = -1
        best_improvement = 0.0        
        for i in range(len(term_name)):
            if i in found_indices: continue 
            current_cols = found_cols + [term_valv[i]]
            A_test = np.column_stack(current_cols)
            coeffs, resid, _, _ = np.linalg.lstsq(A_test, y, rcond=None)
            
            if resid.size > 0:
                y_pred = A_test @ coeffs
                new_error = mse(y - y_pred)
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
        current_error = mse(y - y_pred)
        if current_error < 1e-9: break
    if not found_indices: return [], []
    
    A_final = np.column_stack(found_cols)
    final_coeffs, _, _, _ = np.linalg.lstsq(A_final, y, rcond=None)
    
    return final_coeffs, [term_name[i] for i in found_indices]

coeffs, terms = greedy_algo(x_data, y_data)

parts = [f"{c:.3f}[{t}]" for c, t in zip(coeffs, terms)]
print(" + ".join(parts))

