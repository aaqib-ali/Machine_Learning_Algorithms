### Import Libraries

from Support_Vector_Machine import *

### Create Sample Data for SVM two classes with sample data points

data_dict = {-1:np.array([[1,7],[2,8],[3,8]]), 1:np.array([[5,1],[6,-1],[7,3]])}


svm = Support_Vector_Machine()
svm.SVM_Fit(data=data_dict)
#svm.visualize(data_dict)

### Try Some Sample Valuesfor prediction

predict_sample =[[0,10],[1,3],[3,4],[3,5],[5,5],[5,6],[6,-5],[5,8]]

for p in predict_sample:
    svm.SVM_Predict(p)

svm.visualize(data_dict)