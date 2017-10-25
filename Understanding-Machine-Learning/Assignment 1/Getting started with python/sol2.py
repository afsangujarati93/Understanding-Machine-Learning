# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 18:19:16 2017

@author: afsan
"""
import numpy as np, time

#SECTION 2
def forever ():
    #freezes at 500
    size = 250
    mat1 = np.random.rand(size,size)
    mat2 = np.random.rand(size,size)
    
    start_time = time.time()
    result = []
    for i in range(size):
        emptyarray = [0] * size
        result.append(emptyarray)
        
    # iterate through rows of X
    for i in range(len(mat1)):
        # iterate through columns of Y
        for j in range(len(mat2[0])):
            # iterate through rows of Y
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]
    
    end_time = time.time();
    forever_actual_time = end_time - start_time;
    print('Forever Actual Time:', forever_actual_time)
    return forever_actual_time
    
forever_actual_time = forever()

def mat_fast ():
    size = 500
    mat1 = np.random.rand(size,size)
    mat2 = np.random.rand(size,size)
    start_time = time.time()
    
    np.dot(mat1,mat2)
    
    end_time = time.time();
    mat_fast_actual_time = end_time - start_time;
    print('Mat Fast Actual Time:', mat_fast_actual_time)
    return mat_fast_actual_time
    
mat_fast_actual_time = mat_fast()

lol_actual_diff = forever_actual_time - mat_fast_actual_time 
print('Difference between user created mul matrix and numpy output:', lol_actual_diff)