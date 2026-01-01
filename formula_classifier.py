import numpy as np
from CurveGen_dataset import InputData
_,dimensions=InputData()
formula_library=[{(0.0, 1.0), (0.0,1.0,-1.0), (0.0, 0.0, 1.0)},
                 {(0.0, 1.0,-1.0), (0.0,1.0,-2.0), (0.0, 0.0, 1.0)}]
formula_text=['v=d/t','a=v/t']

for i in range(len(formula_library)):
    if formula_library[i]==dimensions:
        possible_formula_found=formula_text[i]
        print(f'formula detected is {possible_formula_found}')
