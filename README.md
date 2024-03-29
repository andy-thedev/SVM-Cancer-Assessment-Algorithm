# Cancer Assessment Algorithm

## Credit: https://github.com/techwithtim

A repository containing a breast cancer assessment algorithm utilizing SVM. 

## Intro

The model takes patients' breast cancer characteristics, such as mean radius, mean texture, mean perimeter, mean area, etc. to determine if the cancer is malignant or benign.

Language: Python  
Libraries: sklearn  
Data: innate sklearn dataset

## /
**svm.py:**  
The main algorithm (See section: "Design description" below)

## Design description

1) Divides loaded cancer dataset into training(80%) and testing(20%) data

2) Creates svm classifier, with the # of outliers determined in the soft margin = 2 (Soft margins allow some outliers to not be counted for points in border selection)

3) Fits the border deciding malignant or benign in the classifier, utilizing the training data

4) Makes a prediction (malignant or benign) with the testing data

## Outcome

Testing accuracy achieved: 93.86%
