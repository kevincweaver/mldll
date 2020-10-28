#! python
# functions are wrapped in mldll.py and implemented in cpp

import numpy as np
import mldll

matrix1 = np.full((1000,1000), 2, dtype=np.float64)
matrix2 = np.full((1000,1000), 2, dtype=np.float64)

# transposes a matrix 
mldll.transp(matrix1)

# transposes a blocked matrix, ie one submatrix at a time such that
# the processor cache can be better utilized
mldll.transp2(matrix1)

# multiplies two matricies
mldll.mmult(matrix1, matrix2)

# multiplies two matricies, but utilizes transp2() to transform
# the second matrix beforehand to maximise sequential (column) memory accesses
mldll.mmult2(matrix1, matrix2)

# multiplies two matricies in a blocked manner
mldll.mmult3(matrix1, matrix2)

# multiplies two matricies utilizing transp2() and a blocking strategy
mldll.mmult4(matrix1, matrix2)

''' SAMPLE OUTPUT:
[[2. 2. 2. ... 2. 2. 2.]
 [2. 2. 2. ... 2. 2. 2.]
 [2. 2. 2. ... 2. 2. 2.]
 ...
 [2. 2. 2. ... 2. 2. 2.]
 [2. 2. 2. ... 2. 2. 2.]
 [2. 2. 2. ... 2. 2. 2.]]
transp() executed in:  0.031241178512573242 seconds
[[2. 2. 2. ... 2. 2. 2.]
 [2. 2. 2. ... 2. 2. 2.]
 [2. 2. 2. ... 2. 2. 2.]
 ...
 [2. 2. 2. ... 2. 2. 2.]
 [2. 2. 2. ... 2. 2. 2.]
 [2. 2. 2. ... 2. 2. 2.]]
transp2() executed in:  0.03124403953552246 seconds
[[4000. 4000. 4000. ... 4000. 4000. 4000.]
 [4000. 4000. 4000. ... 4000. 4000. 4000.]
 [4000. 4000. 4000. ... 4000. 4000. 4000.]
 ...
 [4000. 4000. 4000. ... 4000. 4000. 4000.]
 [4000. 4000. 4000. ... 4000. 4000. 4000.]
 [4000. 4000. 4000. ... 4000. 4000. 4000.]]
mmult() executed in:  2.203826665878296 seconds
[[4000. 4000. 4000. ... 4000. 4000. 4000.]
 [4000. 4000. 4000. ... 4000. 4000. 4000.]
 [4000. 4000. 4000. ... 4000. 4000. 4000.]
 ...
 [4000. 4000. 4000. ... 4000. 4000. 4000.]
 [4000. 4000. 4000. ... 4000. 4000. 4000.]
 [4000. 4000. 4000. ... 4000. 4000. 4000.]]
mmult2() executed in:  1.3902983665466309 seconds
[[4000. 4000. 4000. ... 4000. 4000. 4000.]
 [4000. 4000. 4000. ... 4000. 4000. 4000.]
 [4000. 4000. 4000. ... 4000. 4000. 4000.]
 ...
 [4000. 4000. 4000. ... 4000. 4000. 4000.]
 [4000. 4000. 4000. ... 4000. 4000. 4000.]
 [4000. 4000. 4000. ... 4000. 4000. 4000.]]
mmult3() executed in:  1.4941833019256592 seconds
[[4000. 4000. 4000. ... 4000. 4000. 4000.]
 [4000. 4000. 4000. ... 4000. 4000. 4000.]
 [4000. 4000. 4000. ... 4000. 4000. 4000.]
 ...
 [4000. 4000. 4000. ... 4000. 4000. 4000.]
 [4000. 4000. 4000. ... 4000. 4000. 4000.]
 [4000. 4000. 4000. ... 4000. 4000. 4000.]]
mmult4() executed in:  1.257746696472168 seconds
'''
