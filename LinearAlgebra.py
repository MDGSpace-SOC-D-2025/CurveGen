import numpy as np
from CurveGen_dataset import test_data,build_baseTerms



def solveSindy(x, y):
   term_name,term_valv=build_baseTerms(x)

   A = np.column_stack(term_valv)

   term_no = A.shape[1]
   magnitudes = np.zeros(term_no)

   for i in range(term_no):
       col_normalise = np.linalg.norm(A[:, i])
       if col_normalise == 0: col_normalise = 1.0
       magnitudes[i] = col_normalise

   A_normalise = A / magnitudes
   term_indices = list(range(len(term_name)))

   for c in range(10):
       A_survive = A_normalise[:, term_indices]
       coeffs_norm, _, _, _ = np.linalg.lstsq(A_survive, y, rcond=None)
       max_coeff = np.max(np.abs(coeffs_norm))
       threshold = max_coeff * 0.5
       surviving_indices = []
       surviving_coeffs_normal = []

       for i, val in enumerate(coeffs_norm):
           if abs(val) > threshold:
               original_idx = term_indices[i]
               surviving_indices.append(original_idx)
               surviving_coeffs_normal.append(val)

       if len(surviving_indices) == len(term_indices):
           final_real_coeffs = []
           for i, idx in enumerate(surviving_indices):
               w_normal = surviving_coeffs_normal[i]
               w_real = w_normal / magnitudes[idx]
               final_real_coeffs.append(w_real)
           return final_real_coeffs, [term_name[i] for i in surviving_indices]

       term_indices = surviving_indices

   return [], []

x_data, y_data = test_data(noise=0.01)
coeffs, terms = solveSindy(x_data, y_data)

parts = [f"{c:.3f}[{t}]" for c, t in zip(coeffs, terms)]
print(" + ".join(parts))







