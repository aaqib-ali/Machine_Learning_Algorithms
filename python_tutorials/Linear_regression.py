#import Libraries

import numpy as np
import matplotlib.pyplot as plt

# Function for Plot the Regression Line
def plot_regression_line(x, y, y_prediction):
    plt.scatter(x,y, color="blue", label="Orignal Values")  # Plot X and Y point
    plt.plot(x,y_prediction, color= "green", label="Predicted Values")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

# Function for Estimating the Coeficients
def Learn_Simple_Linreg(Matirx_A):
    x = Matirx_A[:,0] # Take the X values from the Matrix A
    y = Matrix_A[:,1] # Take the Y values from the Matrix A
    #print(x)
    #print(y)
    mean_x = x.mean()
    mean_y = y.mean()
    xn = x-mean_x
    yn = y-mean_y
    #print(xn)
    #print(yn)
    num = (xn * yn).sum()
    #print(num)
    dnum = np.square(xn).sum()
    #print(dnum)
    beta_1 = num / dnum
    beta_0 = mean_y - (beta_1*mean_x)
    return beta_0, beta_1

# Function for Estimating the Y prediction
def Predict_Simple_Linreg(x,Beta_0, Beta_1):
    y_pred = Beta_0 + (Beta_1 * x)
    return y_pred

# Function for Estimating Coeficients by using Numpy
def Numpy_Linalg(x,y):
    identity = np.ones(len(x))
    #print(identity)
    Identity_transpose = np.vstack([x, identity]).transpose()
    #print(Identity_transpose)
    #print(y)
    np_Beta_1, np_Beta_0 = np.linalg.lstsq(Identity_transpose,y)[0]
    # print("Beta one ",npBeta_1)
    # print("Beta Zero ",npBeta_0)
    return np_Beta_0, np_Beta_1


if __name__ == '__main__':
    # intialize Variables

    n = 100  # Number of Rows for a Matrix
    m = 2  # Number of Column for a Matrix
    mu = 2  # Mu for the vector
    sigma = 0.01  # Sigma for the Vector

    Matrix_A = np.random.normal(mu, sigma,size=(n, m))  # Create Matrix with Size nXm and intiliaze with normal distribution
    learning = Learn_Simple_Linreg(Matrix_A)
    print("Value of Beta_0 using Learn Simple Linreg :",learning[0])
    print("Value of Beta_1 using Learn Simple Linreg  :", learning[1])
    x= Matrix_A[:, 0]
    y = Matrix_A[:, 1]
    y_prediction = Predict_Simple_Linreg(x,learning[0],learning[1])
    #print(y_prediction)
    plot_regression_line(x, y, y_prediction)
    Numpy_Coef = Numpy_Linalg(x,y)
    print("Value of beta_0 using Numpy :",Numpy_Coef[0])
    print("Value of beta_1 using Numpy :",Numpy_Coef[1])


