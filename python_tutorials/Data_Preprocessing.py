#Import Libraries

import pandas as pd

# Function "data_distribution" distribute the data into train and test data by 80:20 respectively
def data_distribution(airline_data,red_wine_data,white_wine_data):
    airline_train = airline_data.sample(frac=0.8, random_state=200)  # Data Distribution on the basis of Random Sampleing making sure Train and Sample Comes Randomly
    red_wine_train = red_wine_data.sample(frac=0.8, random_state=200)
    white_wine_train = white_wine_data.sample(frac=0.8, random_state=200)
    airline_test = airline_data.drop(airline_train.index)
    red_wine_test = red_wine_data.drop(red_wine_train.index)
    white_wine_test = white_wine_data.drop(white_wine_train.index)
    return airline_train,airline_test,red_wine_train,red_wine_test,white_wine_train,white_wine_test

# Main Function
def main():
    # Change Path of the files according to your Machine
    path,path1,path2 = "E:\Workspace\Learning\ML_lab_3\Data_source/Airlines_Data.txt",\
                       "E:\Workspace\Learning\ML_lab_3\Data_source/red_Wine.csv",\
                       "E:\Workspace\Learning\ML_lab_3\Data_source/white_Wine.csv"
    read_airline_file = pd.read_csv(path, header=None, delim_whitespace=True, engine='python')  # Load Airline Data from the file
    read_Red_wine_file = pd.read_csv(path1, header=0, sep=";", engine='python')  # Load Red Wine Data from the file
    read_white_wine_file = pd.read_csv(path2, header=0, sep=";", engine='python')  # Load White Wine Data from the file
    airline_data = pd.DataFrame(read_airline_file).dropna()  # Cast Airline data into another dataframe by Dropping NA for operations
    red_wine_data = pd.DataFrame(read_Red_wine_file).dropna()  # Cast Red Wine data into another dataframe by Dropping NA for operations
    white_wine_data = pd.DataFrame(read_white_wine_file).dropna()  # Cast White Wine data into another dataframe by Dropping NA for operations
    airline_data = pd.get_dummies(airline_data)  # Hot One Encoding Principle only applicable for Airline data, Wine data will not used this Principle
    airline_train,airline_test,red_wine_train,red_wine_test,white_wine_train,white_wine_test = data_distribution(airline_data,red_wine_data,white_wine_data)
    print("Airline Train Data \n",airline_train)
    print("Airline Test Data \n",airline_test)
    print("Red Wine Train Data \n",red_wine_train)
    print("Red Wine Test Data \n",red_wine_test)
    print("White Wine Train Data \n",white_wine_train)
    print("White Wine Test Data \n",white_wine_test)




# Starting point of the Code
if __name__ == '__main__':
    main()