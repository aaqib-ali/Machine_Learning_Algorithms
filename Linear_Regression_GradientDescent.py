#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function "data_distribution" distribute the data into train and test data by 80:20 respectively
def data_distribution(X_white_wine_data, Y_white_wine_data):
    chunk_size = np.random.rand(len(X_white_wine_data)) < 0.8  # Random Sampling till 80 percent
    X_train = X_white_wine_data[chunk_size]  # Assign 80 percent train data for x_train
    X_test = X_white_wine_data[~chunk_size]  # Assign 20 percent test data for x_test
    Y_train = Y_white_wine_data[chunk_size]  # Assign 80 percent train data for Y_train
    Y_test = Y_white_wine_data[~chunk_size]  # Assign 20 percent test data for Y_train
    return X_train, X_test, Y_train, Y_test


def Prediction_Train(Beta, X_train_red_wine):
    Y_hat = np.zeros(len(X_train_red_wine))
    for x in range(len(X_train_red_wine)):
        Y_hat[x] = np.sum(np.matmul(X_train_red_wine.iloc[x, :].tolist(), Beta))
        #print(Y_hat[x])
    return Y_hat


def Prediction_Test(Beta, X_test_red_wine):
    Y_hat = np.zeros(len(X_test_red_wine))
    for x in range(len(X_test_red_wine)):
        Y_hat[x] = np.sum(np.matmul(X_test_red_wine.iloc[x, :].tolist(), Beta))
        #print(Y_hat[x])
    return Y_hat


def Compute_RMSE(residual):
    return np.sqrt(np.square(residual).sum() / len(residual))


def Estimate_Red_Wine_Data_Coefficients_Gradient_Descent(X_train, X_test, Alpha, Epochs, Beta,Y_train,Y_test):
    X_transpose = np.transpose(X_train)
    Beta_transpose = np.transpose(Beta)
    threshold = 0.01  # Stopping Condition
    Least_square = []
    #print(Beta_transpose)
    for iteration in range(Epochs):
        XBeta = np.matmul(X_train, Beta_transpose)
        subtract = Y_train - XBeta
        Dot_prod = np.matmul(X_transpose, subtract)
        diff = np.multiply(-2, Dot_prod)
        Beta = Beta - np.multiply(Alpha, diff)
        Y_hat_train = Prediction_Train(Beta, X_train)
        Y_hat_test = Prediction_Test(Beta, X_test)
        #print(Y_hat_test)
        Error_Least_square = np.subtract(Y_train, Y_hat_train)
        Least_square.append(np.sum(np.square(Error_Least_square)))
        #print(Least_square)
        residual = Y_test-Y_hat_test
        RMSE = Compute_RMSE(residual)
        #print("Least Square -1",Least_square[iteration-1])
        #print("Least Square 1",Least_square[iteration])
        #if(np.abs(Least_square[iteration-1]-Least_square[iteration]<threshold)):
            #print("Converage")
            #break
        print(RMSE)




def main():  # Main Function
    Epochs = 100  # Number Of Iterations
    Alpha = 0.000000001  # Learning Rate
    # Change Path of the files according to your Machine
    path = "E:\Workspace\Learning\ML_lab_3\Data_source/white_Wine.csv"
    read_white_wine_file = pd.read_csv(path, header=0, sep=";", engine='python')  # Load White Wine Data from the file
    X_white_wine_data = pd.DataFrame(read_white_wine_file).dropna()  # Cast White Wine data into another dataframe by Dropping NA for operations
    Y_white_wine_data = X_white_wine_data.iloc[:,11]  # Seprate Quality from the dataSet
    X_white_wine_data.insert(loc=0, column="bias", value=np.ones(len(X_white_wine_data)))  # Adding Bias Column
    X_train, X_test, Y_train, Y_test = data_distribution(X_white_wine_data, Y_white_wine_data)  # Call for Data Distribution
    X_train = X_train.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    X_test = X_test.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    Beta = Beta = np.zeros(len(X_train.columns))  # Create Intial Betas
    #print(Y_train)
    Estimate_Red_Wine_Data_Coefficients_Gradient_Descent(X_train, X_test, Alpha, Epochs, Beta,Y_train,Y_test)
    #print(X_test)




# Starting point of the Code
if __name__ == '__main__':
    main()