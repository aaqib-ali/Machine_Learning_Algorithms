#Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Function "Solve_SLE_Gaussian_elemination" compute the Beta Form Gaussian Elimination methods
def Solve_SLE_Gaussian_elemination(A, b):
    return np.linalg.solve(A,b)

# Function "Y_prediction_Gaussian" compute the prediction for Gaussian Method
def Y_prediction_Gaussian(X_test):
    y_hat_Gaussian = list()
    for x in range(len(X_test)):
        y_hat_Gaussian.append(Beta_Gaussain[0] + X_test.iloc[x, 1] * Beta_Gaussain[1] + X_test.iloc[x, 2] * Beta_Gaussain[2] +
                     X_test.iloc[x, 3] * Beta_Gaussain[3]
                     + X_test.iloc[x, 4] * Beta_Gaussain[4] + X_test.iloc[x, 5] * Beta_Gaussain[5] + X_test.iloc[x, 6] *
                     Beta_Gaussain[6])  # Only Take the columns of X_test other then the bais Column
    return y_hat_Gaussian

# Function "Y_prediction_Cholesky" compute the prediction for Cholesky Decomposition methods
def Y_prediction_Cholesky(X_test):
    y_hat_Cholesky = list()
    for x in range(len(X_test)):
        y_hat_Cholesky.append(
            Beta_Cholesky[0] + X_test.iloc[x, 1] * Beta_Cholesky[1] + X_test.iloc[x, 2] * Beta_Cholesky[2] +
            X_test.iloc[x, 3] * Beta_Cholesky[3]
            + X_test.iloc[x, 4] * Beta_Cholesky[4] + X_test.iloc[x, 5] * Beta_Cholesky[5] + X_test.iloc[x, 6] *
            Beta_Cholesky[6])  # Only Take the columns of X_test other then the bais Column
    return y_hat_Cholesky

# Function "Y_prediction_QR_decomposition" compute the prediction for QR Decomposition method
def Y_prediction_QR_decomposition(X_test):
    y_hat_QR = list()
    for x in range(len(X_test)):
        y_hat_QR.append(
            Beta_Cholesky[0] + X_test.iloc[x, 1] * Beta_QR[1] + X_test.iloc[x, 2] * Beta_QR[2] +
            X_test.iloc[x, 3] * Beta_QR[3]
            + X_test.iloc[x, 4] * Beta_QR[4] + X_test.iloc[x, 5] * Beta_QR[5] + X_test.iloc[x, 6] *
            Beta_QR[6])  # Only Take the columns of X_test other then the bais Column
    return y_hat_QR

# Function "Solve_SLE_QR_decomposition" compute the Beta Form QR Decomposition methods
def Solve_SLE_Cholesky_Decomposition(A, b):
    L = np.linalg.cholesky(A)
    L_transpose = np.transpose(L)
    L_Transpose_inverse = np.linalg.inv(L_transpose)
    L_inverse = np.linalg.inv(L)
    y = np.matmul(L_inverse,b)
    x = np.matmul(L_Transpose_inverse,y)
    return x

# Function "Solve_SLE_QR_decomposition" compute the Beta Form QR Decomposition methods
def Solve_SLE_QR_decomposition(A,b):
    q,r = np.linalg.qr(A)
    q_transpose = np.transpose(q)
    p = np.matmul(q_transpose,b)
    r_inverse = np.linalg.inv(r)
    x = np.matmul(r_inverse,p)
    return x

# Function "Learn_Linreg_NormEq" compute the Beta Form Three methods
def Learn_Linreg_NormEq(X,Y):
    X_transpose = np.transpose(X)
    A = np.dot(X_transpose,X)
    b = np.dot(X_transpose,Y)
    Beta_Gaussian = Solve_SLE_Gaussian_elemination(A,b)  # Call for Gaussian Elimination Method
    Beta_Cholesky = Solve_SLE_Cholesky_Decomposition(A,b)  # Call for Cholesky Decomposition Method
    Beta_QA_decomposition = Solve_SLE_QR_decomposition(A,b)  # Call for QR Decomposition Method
    return Beta_Gaussian,Beta_Cholesky,Beta_QA_decomposition

# Function "data_distribution" distribute the data into train and test data by 80:20 respectively
def data_distribution(relavent_columns, Y_data):
    chunk_size = np.random.rand(len(relavent_columns)) < 0.8  # Random Sampling till 80 percent
    X_train = relavent_columns[chunk_size]  # Assign 80 percent train data for x_train
    X_test = relavent_columns[~chunk_size]  # Assign 20 percent test data for x_test
    Y_train = Y_data[chunk_size]  # Assign 80 percent train data for Y_train
    Y_test = Y_data[~chunk_size]  # Assign 20 percent test data for Y_train
    return X_train,X_test,Y_train,Y_test

# Function "Residual_Gaussian" compute the Residual of Gaussian Predictions
def Residual_Gaussian(Y_test, Y_hat_Gaussian):
    residual = Y_test - Y_hat_Gaussian  # Residual Formula
    plt.title("Residual of Gaussian Elimination Method")
    plt.scatter(residual,Y_test, color="Red")
    plt.xlabel("Residual")
    plt.ylabel("Y_Test")
    plt.legend()
    plt.show()
    return residual

# Function "Residual_Cholesky" compute the Residual of Cholesky Decomposition
def Residual_Cholesky(Y_test, Y_hat_Cholesky):
    residual = Y_test - Y_hat_Cholesky  # Residual Formula
    plt.title("Residual of Cholesky Decomposition Method")
    plt.scatter(residual,Y_test, color="green")
    plt.xlabel("Residual")
    plt.ylabel("Y_Test")
    plt.legend()
    plt.show()
    return residual

# Function "Residual_QR" compute the Residual of QR_Decomposition
def Residual_QR(Y_test, Y_hat_QR_dec):
    residual = Y_test - Y_hat_QR_dec
    plt.title("Residual of QR Decomposition Method")
    plt.scatter(residual,Y_test, color="orange")
    plt.xlabel("Residual")
    plt.ylabel("Y_Test")
    plt.legend()
    plt.show()
    return residual

# Function "Compute_RMSE" compute the RMSE for all the three methods
def Compute_RMSE(residual_Gaussian, residual_Cholesky, residual_QR):
    Error_G = np.sqrt(np.square(residual_Gaussian).sum() / len(residual_Gaussian))
    Error_C = np.sqrt(np.square(residual_Cholesky).sum() / len(residual_Cholesky))
    Error_Q = np.sqrt(np.square(residual_QR).sum() / len(residual_QR))
    return Error_G,Error_C,Error_Q

# Starting point ot the Code
if __name__ == '__main__':

    # Change Path of the file according to your Machine
    path = "house.csv"
    read_file = pd.read_csv(path, header=0, engine='python')  # Load Data from the file
    X_data = pd.DataFrame(read_file)  # Cast data into another dataframe for operations
    Y_data = X_data.iloc[:, 7]  # sales Prices of houses
    relavent_columns = X_data.iloc[:, [1, 2, 3, 4, 5, 6]]  # Only pick relevent columns for prediction(Drop the first column because it is just a serial number of the house)
    relavent_columns.insert(loc=0, column="bias", value=np.ones(len(relavent_columns)))  # Adding Bias Column for Operations
    relavent_columns["nbhd"].replace(["nbhd01", "nbhd02", "nbhd03"], [1, 2, 3],
                                     inplace=True)  # Character Encoding / one hot Encoding Principle
    relavent_columns["brick"].replace(["No", "Yes"], [0, 1], inplace=True)  # Character Encoding / one hot Encoding Principle
    X_train, X_test, Y_train, Y_test = data_distribution(relavent_columns, Y_data)  # Call for Data Distribution
    Beta_Gaussain, Beta_Cholesky, Beta_QR = Learn_Linreg_NormEq(X_train, Y_train)
    Y_hat_Gaussian = Y_prediction_Gaussian(X_test)
    Y_hat_Cholesky = Y_prediction_Cholesky(X_test)
    Y_hat_QR_dec = Y_prediction_QR_decomposition(X_test)
    residual_Gaussian = Residual_Gaussian(Y_test, Y_hat_Gaussian)
    residual_Cholesky = Residual_Cholesky(Y_test, Y_hat_Cholesky)
    residual_QR = Residual_QR(Y_test, Y_hat_QR_dec)
    RMSE_Gaussian, RMSE_Cholesky, RMSE_QR = Compute_RMSE(residual_Gaussian, residual_Cholesky, residual_QR)
    print("Average of Gaussian Residual :", np.mean(residual_Gaussian))
    print("Average of Cholesky Residual :", np.mean(residual_Cholesky))
    print("Average of QR Decomposition Residual :", np.mean(residual_QR))
    print("RMSE of Gaussian Elimination :", RMSE_Gaussian)
    print("RMSE of Cholesky Decomposition :", RMSE_Cholesky)
    print("RMSE of QR Decomposition :", RMSE_QR)
    #plt.plot(Y_hat_Gaussian)
    #plt.plot(Y_hat_Cholesky)
    #plt.plot(Y_hat_QR_dec)
    #plt.plot(list(Y_test))
    #plt.show()
    # print(X_test)