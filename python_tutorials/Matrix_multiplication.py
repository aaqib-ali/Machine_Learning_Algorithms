#import Libraries

import numpy as np
import matplotlib.pyplot as plt

#intialize Variables

n=100  # Number of Rows for a Matrix
m=20   # Number of Column for a Matrix
mu = 2 # Mu for the vector
sigma = 0.01   # Sigma for the Vector

Matrix_A = np.random.normal(mu, sigma, size=(n,m))  # Create Matrix with Size nXm with normal distribution
vector_v = np.random.normal(mu, sigma, size=(m,1))

#vector_c = np.matmul(Matrix_A,vector_v)  # Multiplication of matrix using Numpy Library
vector_c = np.zeros((n,1))  # Resultant vector intialize with Zeros
for x in range(0,len(Matrix_A)):   # Iteratively Multiply the Matrix with Vector_V
    for y in range(0,len(vector_v)):
        vector_c[x][0]  = vector_c[x][0] + (Matrix_A[x][y] * vector_v[y][0])

print("Mean of a Vector C :", vector_c.mean()) # Mean Of a vector C
print("Standard Deviation of a Vector C :", vector_c.std()) # Standard Deviation of a Vector C
plt.hist(vector_c, bins=5, color ='Green')  # Plot Histogram for vector C
plt.show()