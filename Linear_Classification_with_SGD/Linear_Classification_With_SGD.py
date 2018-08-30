#Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function Load_Data read the file form the source and precess it further
def Load_Data():
    # Chnage the Path according to the desiredMachine
    path = "E:\Workspace\Learning\ML_Lab_4/bank_additional.csv"
    read_bank_data = pd.read_csv(path, header=0, sep=";", engine='python')  # Load Bank Additional Data
    read_bank_data = read_bank_data.dropna()
    bank_data = pd.get_dummies(read_bank_data)
    return bank_data

# Function Data_Distribution used for Data Split into 80:20 Ratio
def Data_Distribution(read_bank_file):
    bank_data_train = read_bank_file.sample(frac=0.8, random_state=200)  # 80% Ratio
    bank_data_test = read_bank_file.drop(bank_data_train.index)  # 20% Ratio
    return bank_data_train, bank_data_test

# Function Take_labels Extract the Labels from train and test data and add Bias Column for Computations
def Take_labels(bank_data_train, bank_data_test):
    bank_data_train.insert(loc=0, column='bias',value=np.ones(len(bank_data_train)))  # Adding Bias
    bank_data_test.insert(loc=0, column='bias',value=np.ones(len(bank_data_test)))  # Adding Bias

    bank_data_train_labels = bank_data_train.iloc()[:,-1].tolist()  # Taking Labels from Train Data
    bank_data_test_labesl = bank_data_test.iloc()[:,-1].tolist()  # Takiing Labels from Test Data

    del bank_data_train["y_yes"]  # Remove the Unnecessary Columns from train DataSet
    del bank_data_train["y_no"]
    del bank_data_test["y_yes"]  #  Remove the Unnecessary Columns from test DataSet
    del bank_data_test["y_no"]

    return bank_data_train,bank_data_test,bank_data_train_labels,bank_data_test_labesl

# Function STEP_SIZE_ADAGRAD Compute the Step Length by using ADAGRAD
def STEP_SIZE_ADAGRAD(mul,h_array,Alpha_array):
    gradient_product = np.multiply(mul,mul)
    h_array = np.add(h_array,gradient_product)

    for x in range(len(h_array)): # Equation Formula
        if h_array[x] is 0:
            h_array[x]= 0.000001  # Make Sure Division Can't cause undefined
        h_array[x] = Alpha_array[x]/np.sqrt(h_array[x])

    return Alpha_array,h_array

# Function STEP_SIZE_BOLD_DRIVER Compute the Step Length by using BOLD DRIVER
def STEP_SIZE_BOLD_DRIVER(Alpha, Current_Loss, old_Loss, Positive_Alpha, Negative_Alpha):
    if(Current_Loss > old_Loss):
        new_Alpha = Positive_Alpha*Alpha
    else:
        new_Alpha = Negative_Alpha*Alpha
    return new_Alpha

# Function LOG_LOSS Compute the Log Loss of the Test Data Set
def LOG_LOSS(beta, bank_data_test, bank_data_test_labels):
    Loss = 0.0

    for x in range(len(bank_data_test)):
        exponential = 1 / 1 + np.exp(np.multiply(-1, np.matmul(np.transpose(beta), bank_data_test.iloc()[x,:])))
        numinator = bank_data_test_labels[x] * np.log(exponential)
        denuminator = (1 - bank_data_test_labels[x]) * np.log(1 - exponential)
        Loss = Loss+(numinator + denuminator)
        #print("Log Loss", Loss)

    Loss = Loss * (-1)
    return Loss

# Function LEARN_LOGREG_GRADIENT_ASCENT Compute the Gradiant Ascent
def LEARN_LOGREG_GRADIENT_ASCENT(bank_data_train, bank_data_train_labels, Alpha, Epochs,bank_data_test,bank_data_test_labels):
    beta = np.zeros(len(bank_data_train.columns))  # Initial Betas for the Calculations
    threshold = 0.5  # Threshold for the Loss

    Optimaization_choice = 1  # Bold Driver Algorithm Choice / AdaGrad Algorithm Choice (Change Accordingly)

    Negative_Alpha = 0.3
    Positive_Alpha = 1.1

    fill_columns = len(bank_data_train.iloc()[0,:])

    Alpha_array = np.empty(fill_columns)
    Alpha_array.fill(Alpha)

    h_array = np.zeros(fill_columns)

    function_lose = []  # List to save Lose on each Iteration
    log_Loss_test = []  # List to save log Loss on each Iteration
    Current_Loss = 0.0
    for x in range(len(bank_data_train)):  # Compute the Loss of Every Row
        Current_Loss = Current_Loss+ np.multiply(bank_data_train_labels[x],np.dot(bank_data_train.iloc()[x,:].tolist(),beta))-np.log(1+np.exp(np.dot(bank_data_train.iloc()[x,:].tolist(),beta)))

    for x in range(Epochs):
        print("Iteration Number :", x)
        for row in range(len(bank_data_train)):
            Beta_transpose = np.transpose(beta)  # Computing Equation
            power = np.matmul(Beta_transpose, bank_data_train.iloc()[row,:])
            power = np.multiply(-1, power)

            prediction = 1 / (1 + np.exp(power))  # Computing Prediction

            Yp = bank_data_train_labels[row] - prediction

            mul = np.multiply(np.transpose(bank_data_train.iloc()[row,:].tolist()), Yp)  # Compute Gradient

            beta = np.add(beta, np.multiply(Alpha, mul))

        old_Loss = Current_Loss
        Current_Loss = 0.0
        Loss = LOG_LOSS(beta, bank_data_test, bank_data_test_labels)
        log_Loss_test.append(Loss)

        for x in range(len(bank_data_train)):
            Current_Loss = Current_Loss + np.multiply(bank_data_train_labels[x],np.dot(bank_data_train.iloc()[x, :].tolist(), beta)) - np.log(1 + np.exp(np.dot(bank_data_train.iloc()[x, :].tolist(), beta)))
        Loss_didderence = Current_Loss - old_Loss
        function_lose.append(Loss_didderence)
        print("Loss Differnece :",Loss_didderence)
        if(Optimaization_choice == 1 and x>0):  # BOLD DRIVER step Length Choice
            Alpha = STEP_SIZE_BOLD_DRIVER(Alpha,Current_Loss,old_Loss,Positive_Alpha,Negative_Alpha)

        if(Optimaization_choice == 2 and x>0):  # ADAGRAD step Length Choice
           Alpha_array,h = STEP_SIZE_ADAGRAD(mul,h_array,Alpha_array)
           #STEP_SIZE_ADAGRAD(bank_data_train,bank_data_train_labels,h_array,Epochs,Alpha)
        beta = np.add(beta,np.multiply(Alpha_array,mul))  # Update Bets's

        if (Current_Loss - old_Loss < threshold):  # Convergence Condition
            #print(x)
            print("Old Loss", old_Loss)
            print("New Loss", Current_Loss)
            print("The DataSet Converage ")
            break
    return beta,function_lose,log_Loss_test


# Main Function
def main():
    Epochs = 200  # Number of Iterations
    Alpha = 0.000000001  # Step Length
    read_bank_file = Load_Data()  # Load File from the Machine
    bank_data_train, bank_data_test = Data_Distribution(read_bank_file)  # Data Distribution
    bank_data_train, bank_data_test, bank_data_train_labels, bank_data_test_labels= Take_labels(bank_data_train,bank_data_test)
    Beta,function_loss,log_loss_test = LEARN_LOGREG_GRADIENT_ASCENT(bank_data_train,bank_data_train_labels, Alpha, Epochs,bank_data_test,bank_data_test_labels)
    plt.title("Loss Difference by BOLD DRIVER")
    plt.plot(function_loss)  # Show Function Loss List of the train Data
    plt.show()

    plt.title("Log Loss")
    plt.plot(log_loss_test)  # Show Log Loss of the test Data
    plt.show()
    #print("New Coffiencts ",Beta)


# Starting Point of the Program
if __name__ == '__main__':
    main()