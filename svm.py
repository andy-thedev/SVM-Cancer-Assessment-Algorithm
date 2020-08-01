import sklearn
from sklearn import datasets
from sklearn import svm

# For comparison to KNN model, KNN does not do well with high dimensions in comparison to SVM
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

# SVM, support vector machines uses classification datasets
# Divides dataset linearly into hyperplanes.
# Finds the two closest points from opposite classes, and draws
# a border that has the same distance from the two selected points.
# Such border can be made in infinitely many ways, so the best selection
# is a border that has the longest distance to the opposite points to minimize error.
# We can use soft margins to allow some outliers to not be counted for points in border selection

# We use kernels (a function(x1, x2) -> x3) to separate intertwined data

# Load innate sklearn dataset
cancer = datasets.load_breast_cancer()

# Analyze labels and features of dataset
print(cancer.feature_names)
print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

print(x_train, y_train)
classes = ["malignant", "benign"]

# C is the # of outliers that may be determined in the soft margin
clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)

print(acc)
