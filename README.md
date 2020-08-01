A repository containing a breast cancer assessment algorithm utilizing SVM. 

Introduction:

The model takes patients' breast cancer characteristics, such as mean radius, mean texture, mean perimeter, mean area, etc. to determine if the cancer is malignant or benign.

svm.py -> The main algorithm

Design description:

1) Divides loaded cancer dataset into training(80%) and testing(20%) data

2) Creates svm classifier, with the # of outliers determined in the soft margin = 2 (Soft margins allow some outliers to not be counted for points in border selection)

3) Fits the border deciding malignant or benign in the classifier, utilizing the training data

4) Makes a prediction (malignant or benign) with the testing data

Libraries utilized: sklearn

Dataset utilized: innate sklearn dataset

Testing accuracy achieved: 93.86%
