### Import Libraries

import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

### Step 1 : Data Collection
dataset = pd.read_csv('Sample_Data.csv')  ### Read Data from the file
dataset.head(5)

### Step 2 : Data Analysis
sns.countplot(x="Survived", data=dataset)
sns.countplot(x="Survived", hue="Sex", data=dataset)
sns.countplot(x="Survived", hue="Pclass", data=dataset)
dataset["Age"].plot.hist()
dataset.info()

### Step 3 : Data Wrangling (Clean Data)

dataset.isnull().sum()  ### Number of Null Values of each Column
sns.heatmap(dataset.isnull(), yticklabels=False)
sns.boxplot(x="Pclass", y="Age", data=dataset)
dataset.drop("Cabin", axis=1, inplace=True)  ### Drop Unused Column
dataset.head(5)
#print(dataset)
dataset.dropna(inplace=True)  ### Drop NA Values
dataset.isnull().sum()

sex = pd.get_dummies(dataset["Sex"], drop_first=True)
sex.head(3)
#print(sex)
embarked = pd.get_dummies(dataset["Embarked"], drop_first=True)
embarked.head(3)
#print(embarked)
Pclass = pd.get_dummies(dataset["Pclass"], drop_first=True)
Pclass.head(3)
#print(Pclass.head(3))
dataset.drop(['Sex', 'Embarked', 'PassengerId', 'Ticket', 'Name','Pclass'], axis=1, inplace=True)   ### Drop Unused Coloumns
#print(dataset)
dataset = pd.concat([dataset,sex,embarked,Pclass], axis=1)
print(dataset.head())

### Step 3 : Training Data and testing Data

x = dataset.drop(['Survived'], axis=1)  ### Features
y = dataset['Survived']   ### Labels
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)   ### Fit Our Model

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))   ### Classification Report
print(confusion_matrix(y_test, predictions))   ### Confusion Metrics
print(accuracy_score(y_test, predictions))   ### Accuracy
